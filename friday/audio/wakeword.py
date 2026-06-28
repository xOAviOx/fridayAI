"""Phase 7 — Always-on wake word detection + VAD-based recording.

Replaces push-to-talk with a hands-free flow:

1. ``WakeWordListener`` holds a single continuous mic stream at 16 kHz.
   A sounddevice *callback* fires every 80 ms (1 280 samples) and puts
   audio into a queue — the same pattern openwakeword's own examples use.
2. The main thread drains the queue and scores each chunk with
   openwakeword.  When the score exceeds ``sensitivity`` the listener
   enters *recording mode*.
3. In recording mode chunks are still read from the same queue.  Each
   1 280-sample chunk is split into 480-sample sub-frames for webrtcvad.
   Recording ends when ``silence_ms`` of consecutive silence is detected
   or ``max_record_s`` seconds elapse.
4. The utterance (float32 numpy array) is passed to the ``on_wake``
   callback — identical to what MicRecorder produced, so the turn
   pipeline is unchanged.

PTT fallback
------------
``ptt_start()`` / ``ptt_stop()`` inject manual start/stop that bypasses
wake-word scoring.  The same queue supplies audio, so no second stream
opens.  Ctrl+Space still works alongside always-on listening.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

_OWW_CHUNK_SAMPLES = 1_280   # 80 ms @ 16 kHz — openwakeword's native chunk
_OWW_CHUNK_MS      = 80
_VAD_FRAME_SAMPLES = 480     # 30 ms @ 16 kHz — webrtcvad frame size
_VAD_FRAME_MS      = 30


class WakeWordListener:
    """Always-on mic listener: wake word → VAD utterance → callback.

    Parameters
    ----------
    on_wake:
        Called on the listener thread with a float32 mono numpy array of
        the captured utterance (everything after the wake phrase).
    model_name:
        openwakeword model to load.  Built-in ONNX choices: ``hey_jarvis``,
        ``alexa``, ``hey_mycroft``, ``hey_rhasspy``.
    sensitivity:
        Score threshold 0–1.  Start low (0.1) and raise if false positives
        are a problem.
    vad_aggressiveness:
        webrtcvad aggressiveness 0–3.
    silence_ms:
        Consecutive milliseconds of silence that end an utterance.
    max_record_s:
        Hard cap on utterance length.
    sample_rate:
        Must be 16 000 Hz.
    """

    def __init__(
        self,
        *,
        on_wake: Callable[[Any], None],
        model_name: str = "hey_jarvis",
        sensitivity: float = 0.1,
        vad_aggressiveness: int = 2,
        silence_ms: int = 900,
        max_record_s: float = 15.0,
        sample_rate: int = 16_000,
    ) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "WakeWordListener requires sounddevice + numpy.\n"
                "    pip install -e '.[audio]'"
            ) from exc
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "WakeWordListener requires webrtcvad.\n"
                "    pip install -e '.[wake-word]'"
            ) from exc
        try:
            from openwakeword.model import Model as _OWW
        except ImportError as exc:
            raise RuntimeError(
                "WakeWordListener requires openwakeword.\n"
                "    pip install -e '.[wake-word]'"
            ) from exc

        self._np  = np
        self._sd  = sd
        self._vad = webrtcvad.Vad(vad_aggressiveness)

        self._model_name   = model_name
        self._sensitivity  = sensitivity
        self._silence_ms   = silence_ms
        self._max_record_s = max_record_s
        self._sample_rate  = sample_rate
        self._on_wake      = on_wake

        self._stop_event  = threading.Event()
        self._ptt_active  = threading.Event()
        self._ptt_release = threading.Event()
        self._thread: threading.Thread | None = None

        # Single shared queue — the callback always fills it, the main
        # thread always drains it regardless of detection vs recording mode.
        self._q: queue.Queue[Any] = queue.Queue(maxsize=100)

        # Download and load model.
        log.info("loading openwakeword model %r …", model_name)
        try:
            import openwakeword as _oww_pkg
            _oww_pkg.utils.download_models()
        except Exception:
            pass   # already cached or offline — model load below will catch errors

        self._oww = _OWW(wakeword_models=[model_name], inference_framework="onnx")
        log.info(
            "wake word listener ready — model=%r sensitivity=%.2f silence=%dms max=%.0fs",
            model_name, sensitivity, silence_ms, max_record_s,
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
        self._ptt_release.set()   # unblock any waiting PTT read
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ---- PTT override -------------------------------------------------------

    def ptt_start(self) -> None:
        """Bypass wake-word scoring and enter recording mode now."""
        self._ptt_release.clear()
        self._ptt_active.set()

    def ptt_stop(self) -> None:
        """End PTT recording; utterance delivered via on_wake callback."""
        self._ptt_active.clear()
        self._ptt_release.set()

    # ---- main loop ----------------------------------------------------------

    def _run(self) -> None:
        np = self._np
        sd = self._sd

        log.info(
            "always-on listener active — say %r or press PTT to activate",
            self._model_name,
        )

        def _mic_callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            """Called by PortAudio every _OWW_CHUNK_SAMPLES frames."""
            if status:
                log.debug("mic callback status: %s", status)
            try:
                self._q.put_nowait(indata.copy())
            except queue.Full:
                pass   # drop oldest chunk rather than block the audio thread

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=_OWW_CHUNK_SAMPLES,
                callback=_mic_callback,
            ):
                while not self._stop_event.is_set():

                    # ---- PTT override ----------------------------------------
                    if self._ptt_active.is_set():
                        audio = self._record_ptt()
                        self._deliver(audio)
                        continue

                    # ---- wake-word detection ---------------------------------
                    try:
                        chunk = self._q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    chunk_1d = chunk[:, 0]   # (1280, 1) → (1280,)
                    score = self._score(chunk_1d)

                    if score >= 0.01:
                        log.info(
                            "wakeword score: %.4f  threshold=%.2f  %s",
                            score, self._sensitivity,
                            "█" * int(score * 40),
                        )

                    if score >= self._sensitivity:
                        log.info(
                            "wake word %r DETECTED (score=%.3f) — recording utterance",
                            self._model_name, score,
                        )
                        self._oww.reset()
                        audio = self._record_vad()
                        self._deliver(audio)

        except Exception:
            log.exception("wake word listener crashed")

    # ---- scoring ------------------------------------------------------------

    def _score(self, chunk_int16: Any) -> float:
        """Feed one int16 chunk to openwakeword, return the max model score."""
        pred: dict = self._oww.predict(chunk_int16)
        if not pred:
            return 0.0
        return float(max(pred.values()))

    # ---- recording: VAD mode ------------------------------------------------

    def _record_vad(self) -> Any:
        """Record from the queue until silence or max_record_s."""
        np = self._np
        # How many 80ms chunks = silence_ms of silence?
        silence_chunks_needed = max(1, self._silence_ms // _OWW_CHUNK_MS)
        max_chunks = int(self._max_record_s * 1_000 / _OWW_CHUNK_MS)

        frames: list[Any] = []
        consecutive_silence = 0
        speech_started = False

        for _ in range(max_chunks):
            if self._stop_event.is_set() or self._ptt_active.is_set():
                break

            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                break

            chunk_1d = chunk[:, 0]
            frames.append(chunk_1d.copy())

            # VAD: split 1280-sample chunk into 480-sample sub-frames.
            is_speech_in_chunk = False
            for i in range(0, _OWW_CHUNK_SAMPLES - _VAD_FRAME_SAMPLES + 1, _VAD_FRAME_SAMPLES):
                sub = chunk_1d[i : i + _VAD_FRAME_SAMPLES]
                if len(sub) == _VAD_FRAME_SAMPLES:
                    try:
                        if self._vad.is_speech(sub.tobytes(), self._sample_rate):
                            is_speech_in_chunk = True
                            break
                    except Exception:
                        is_speech_in_chunk = True

            if is_speech_in_chunk:
                speech_started = True
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            if speech_started and consecutive_silence >= silence_chunks_needed:
                log.debug("silence detected — ending utterance")
                break

        if not frames:
            return np.zeros(0, dtype=np.float32)

        # Trim trailing silence — keep one chunk for a natural cutoff.
        keep = max(1, len(frames) - silence_chunks_needed + 1)
        audio_i16 = np.concatenate(frames[:keep])
        audio_f32 = audio_i16.astype(np.float32) / 32_768.0
        log.info("utterance recorded: %.2fs (VAD mode)", len(audio_f32) / self._sample_rate)
        return audio_f32

    # ---- recording: PTT mode ------------------------------------------------

    def _record_ptt(self) -> Any:
        """Record from the queue until PTT is released or max_record_s."""
        np = self._np
        max_chunks = int(self._max_record_s * 1_000 / _OWW_CHUNK_MS)
        frames: list[Any] = []

        for _ in range(max_chunks):
            if self._stop_event.is_set() or self._ptt_release.is_set():
                break
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            frames.append(chunk[:, 0].copy())

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
