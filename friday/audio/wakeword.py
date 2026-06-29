"""Phase 7 — Always-on VAD + Whisper wake word detection.

Why not openwakeword?
    The pretrained openwakeword models (hey_jarvis, alexa, etc.) were
    trained primarily on American English.  They score near 0 for many
    non-American accents, so the threshold can never be met.

This implementation uses a two-stage pipeline that works for any accent:

1. **webrtcvad** listens for speech on-device with zero latency and zero
   API cost.  Only frames that contain actual speech pass stage 2.
2. **Groq Whisper** transcribes the first ~1.5 s of each utterance.  If
   the transcript contains the configured wake phrase (or a fuzzy match),
   the listener enters full recording mode.

Flow
----
idle → VAD hears speech → record 1.5 s → Whisper check →
    wake phrase found: record full utterance with VAD end-detection
                       → deliver float32 audio to on_wake callback
    wake phrase absent: discard clip, return to idle

PTT fallback
------------
ptt_start() / ptt_stop() skip VAD+Whisper entirely and go straight to
full utterance recording.  Ctrl+Space still works at any time.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

_FRAME_MS      = 30          # webrtcvad frame size (10 / 20 / 30 ms)
_FRAME_SAMPLES = 480         # 30 ms @ 16 kHz
_QUEUE_TIMEOUT = 0.3


class WakeWordListener:
    """Always-on VAD + Whisper listener → on_wake callback.

    Parameters
    ----------
    on_wake:
        Called with a float32 mono numpy array of the full utterance
        (the part *after* the wake phrase is stripped before delivery).
    transcribe_fn:
        ``transcribe_fn(pcm16_bytes, sample_rate) -> str`` — typically
        wraps ``GroqWhisperSTT.transcribe``.  Used only for the short
        wake-phrase check clip, not the full utterance.
    wake_phrase:
        The phrase to listen for.  Case-insensitive substring match
        against the Whisper transcript.  Defaults to "hey friday";
        common mishearings are also accepted automatically.
    vad_aggressiveness:
        webrtcvad aggressiveness 0–3.
    silence_ms:
        Consecutive silence (ms) that ends a full utterance.
    max_record_s:
        Hard cap on utterance length.
    wake_clip_s:
        How many seconds to record for the wake-phrase check.
    sample_rate:
        Must be 16 000 Hz.
    """

    def __init__(
        self,
        *,
        on_wake: Callable[[Any], None],
        transcribe_fn: Callable[[bytes, int], Any],
        wake_phrase: str = "hey friday",
        vad_aggressiveness: int = 2,
        silence_ms: int = 900,
        max_record_s: float = 15.0,
        wake_clip_s: float = 1.5,
        sample_rate: int = 16_000,
    ) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Requires sounddevice + numpy.") from exc
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "Requires webrtcvad.  Install with: pip install -e '.[wake-word]'"
            ) from exc

        self._np            = np
        self._sd            = sd
        self._vad           = webrtcvad.Vad(vad_aggressiveness)
        self._on_wake       = on_wake
        self._transcribe    = transcribe_fn
        self._wake_phrase   = wake_phrase.lower().strip()
        self._silence_ms    = silence_ms
        self._max_record_s  = max_record_s
        self._wake_clip_s   = wake_clip_s
        self._sample_rate   = sample_rate

        self._stop_event  = threading.Event()
        self._ptt_active  = threading.Event()
        self._ptt_release = threading.Event()
        self._thread: threading.Thread | None = None

        # Audio queue — callback fills it, listener thread drains it.
        self._q: queue.Queue[Any] = queue.Queue(maxsize=200)

        log.info(
            "wake word listener ready — phrase=%r vad=%d silence=%dms",
            wake_phrase, vad_aggressiveness, silence_ms,
        )

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="friday-wakeword"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._ptt_release.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ---- PTT override -------------------------------------------------------

    def ptt_start(self) -> None:
        self._ptt_release.clear()
        self._ptt_active.set()

    def ptt_stop(self) -> None:
        self._ptt_active.clear()
        self._ptt_release.set()

    # ---- main loop ----------------------------------------------------------

    def _run(self) -> None:
        np = self._np
        sd = self._sd

        log.info(
            "always-on listener active — say %r or press PTT to talk",
            self._wake_phrase,
        )

        def _cb(indata: Any, frames: int, _t: Any, status: Any) -> None:
            if status:
                log.debug("mic status: %s", status)
            try:
                self._q.put_nowait(indata.copy())
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=_FRAME_SAMPLES,   # 30 ms frames match webrtcvad exactly
                callback=_cb,
            ):
                while not self._stop_event.is_set():
                    # --- PTT override: skip VAD/Whisper, record immediately ---
                    if self._ptt_active.is_set():
                        audio = self._record_until_ptt_release()
                        self._deliver(audio)
                        continue

                    # --- Stage 1: wait for VAD to detect speech onset --------
                    frame = self._get_frame()
                    if frame is None:
                        continue
                    if not self._is_speech(frame):
                        continue

                    # Speech detected — collect a short clip for wake check.
                    log.debug("speech onset detected — collecting wake clip")
                    clip_frames = [frame]
                    clip_samples = int(self._wake_clip_s * self._sample_rate)
                    clip_frame_count = clip_samples // _FRAME_SAMPLES

                    for _ in range(clip_frame_count - 1):
                        f = self._get_frame()
                        if f is None:
                            break
                        clip_frames.append(f)

                    # --- Stage 2: Whisper wake-phrase check ------------------
                    clip_i16 = np.concatenate([f[:, 0] for f in clip_frames])
                    clip_f32 = clip_i16.astype(np.float32) / 32_768.0

                    from friday.audio.encoding import float32_to_pcm16_bytes
                    try:
                        result = self._transcribe(
                            float32_to_pcm16_bytes(clip_f32),
                            self._sample_rate,
                        )
                        transcript = (result.text or "").lower().strip()
                    except Exception:
                        log.debug("wake clip transcription failed", exc_info=True)
                        continue

                    log.info("wake clip transcript: %r", transcript)

                    if not self._matches_wake_phrase(transcript):
                        log.debug("no wake phrase — discarding clip")
                        continue

                    # --- Wake phrase confirmed — record full utterance -------
                    log.info("wake phrase detected! recording full utterance…")
                    audio = self._record_utterance_vad()
                    self._deliver(audio)

        except Exception:
            log.exception("wake word listener crashed")

    # ---- wake phrase matching -----------------------------------------------

    def _matches_wake_phrase(self, transcript: str) -> bool:
        """True if the transcript contains the wake phrase or a common mishearing."""
        if not transcript:
            return False
        # Accept the configured phrase
        if self._wake_phrase in transcript:
            return True
        # Common Whisper mishearings of "hey friday"
        _aliases = [
            "hey friday", "hey freday", "a friday", "hey fried",
            "hey fryday", "hey, friday", "hey jarvis", "hey jarves",
            "hi friday", "hi jarvis", "hey fry day",
        ]
        for alias in _aliases:
            if alias in transcript:
                return True
        return False

    # ---- VAD helpers --------------------------------------------------------

    def _get_frame(self) -> Any:
        """Pull one 30 ms frame from the queue, or None on timeout/stop."""
        try:
            return self._q.get(timeout=_QUEUE_TIMEOUT)
        except queue.Empty:
            return None

    def _is_speech(self, frame: Any) -> bool:
        try:
            return self._vad.is_speech(frame[:, 0].tobytes(), self._sample_rate)
        except Exception:
            return False

    # ---- utterance recording: VAD silence gate ------------------------------

    def _record_utterance_vad(self) -> Any:
        """Record until silence or max_record_s and return float32 audio.

        Behaviour notes
        ---------------
        * The wake-clip Whisper call takes ~500 ms+, during which the
          audio queue often empties.  ``_get_frame()`` then returns
          ``None`` on its 300 ms internal timeout.  We must NOT exit on
          the first ``None`` — the user is still drawing breath to
          speak the command.  Instead we wait up to ``_grace_s`` for
          speech to start; only after that do empty frames count as
          end-of-stream.
        * Once speech has been heard, normal VAD silence-gate ends the
          recording.
        """
        np = self._np
        silence_frames_needed = max(1, self._silence_ms // _FRAME_MS)
        max_frames = int(self._max_record_s * 1_000 / _FRAME_MS)
        # How long to wait for the user to begin their command after
        # the wake phrase before giving up.  3 s is comfortable.
        grace_s = 3.0
        grace_empty_polls = int(grace_s / _QUEUE_TIMEOUT) + 1

        frames: list[Any] = []
        consecutive_silence = 0
        speech_started = False
        empty_polls = 0

        for _ in range(max_frames):
            if self._stop_event.is_set() or self._ptt_active.is_set():
                break
            frame = self._get_frame()
            if frame is None:
                # No audio chunk arrived this poll.  Tolerate it while
                # we're still waiting for the user to start speaking.
                if speech_started:
                    # Mid-utterance silence shouldn't normally show up
                    # as None (the mic callback keeps feeding the queue
                    # even during quiet), but if it does, treat it like
                    # a silence frame so the silence gate can still end
                    # the recording.
                    consecutive_silence += 1
                    if consecutive_silence >= silence_frames_needed:
                        break
                    continue
                empty_polls += 1
                if empty_polls >= grace_empty_polls:
                    # User never said anything within the grace window.
                    break
                continue

            frames.append(frame[:, 0].copy())

            if self._is_speech(frame):
                speech_started = True
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            if speech_started and consecutive_silence >= silence_frames_needed:
                break

        if not frames:
            return np.zeros(0, dtype=np.float32)

        keep = max(1, len(frames) - silence_frames_needed + 1)
        audio_i16 = np.concatenate(frames[:keep])
        audio_f32 = audio_i16.astype(np.float32) / 32_768.0
        log.info("utterance recorded: %.2fs (VAD mode)", len(audio_f32) / self._sample_rate)
        return audio_f32

    # ---- utterance recording: PTT gate --------------------------------------

    def _record_until_ptt_release(self) -> Any:
        """Record until PTT is released or max_record_s."""
        np = self._np
        max_frames = int(self._max_record_s * 1_000 / _FRAME_MS)
        frames: list[Any] = []

        for _ in range(max_frames):
            if self._stop_event.is_set() or self._ptt_release.is_set():
                break
            frame = self._get_frame()
            if frame is not None:
                frames.append(frame[:, 0].copy())

        self._ptt_release.clear()

        if not frames:
            return np.zeros(0, dtype=np.float32)

        audio_i16 = np.concatenate(frames)
        audio_f32 = audio_i16.astype(np.float32) / 32_768.0
        log.info("utterance recorded: %.2fs (PTT mode)", len(audio_f32) / self._sample_rate)
        return audio_f32

    # ---- delivery -----------------------------------------------------------

    def _deliver(self, audio: Any) -> None:
        if audio is None or len(audio) == 0:
            return
        try:
            self._on_wake(audio)
        except Exception:
            log.exception("on_wake callback raised")


__all__ = ["WakeWordListener"]
