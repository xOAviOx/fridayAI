"""Phase 7 — Always-on wake word detection + VAD-based recording.

Replaces push-to-talk with a hands-free flow:

1. ``WakeWordListener`` holds a single continuous mic stream at 16 kHz.
2. 80 ms chunks (1 280 samples) are scored by **openwakeword** against a
   configurable model (default: ``hey_jarvis`` — the closest built-in to
   "Hey Friday").  When the score exceeds ``sensitivity``, the listener
   enters *recording mode*.
3. In recording mode, 30 ms frames (480 samples) are fed to **webrtcvad**.
   Recording ends when ``silence_ms`` of consecutive silence is detected
   or ``max_record_s`` seconds elapse.
4. The utterance (float32 numpy array, same format as ``MicRecorder``) is
   handed to the ``on_wake`` callback — the turn pipeline is unchanged.

PTT fallback
------------
``WakeWordListener`` also owns the PTT path when it is active, so a
second mic stream never opens.  ``ptt_start()`` / ``ptt_stop()`` inject a
manual "start / stop recording" trigger that bypasses wake-word scoring.
This lets the user press Ctrl+Space at the desk and say "hey friday" from
the couch — both paths produce the same callback.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

# openwakeword expects 80 ms chunks at 16 kHz = 1 280 samples.
_OWW_CHUNK_SAMPLES = 1_280      # 80 ms @ 16 kHz
_OWW_CHUNK_MS      = 80

# webrtcvad works on 10 / 20 / 30 ms frames.  30 ms is the most
# aggressive silence detection at reasonable CPU cost.
_VAD_FRAME_MS      = 30
_VAD_FRAME_SAMPLES = 480        # 30 ms @ 16 kHz


class WakeWordListener:
    """Always-on mic listener: wake word → VAD utterance → callback.

    Parameters
    ----------
    on_wake:
        Called on the listener thread with a float32 mono numpy array of
        the captured utterance (everything after the wake phrase).
    model_name:
        openwakeword model to load.  Built-in ONNX choices: ``hey_jarvis``,
        ``alexa``, ``hey_mycroft``, ``hey_rhasspy``.  Point to a custom
        ``.onnx`` path for a trained "hey_friday" model.
    sensitivity:
        Score threshold 0–1.  Lower = more triggers (false positives);
        higher = fewer triggers (may miss).  0.5 is a safe start.
    vad_aggressiveness:
        webrtcvad aggressiveness 0–3.  Higher = more frames classified as
        silence (ends recording sooner in noisy environments).
    silence_ms:
        Consecutive milliseconds of silence that end an utterance.
    max_record_s:
        Hard cap on utterance length.
    sample_rate:
        Must be 16 000 — both openwakeword and webrtcvad require 16 kHz.
    """

    def __init__(
        self,
        *,
        on_wake: Callable[[Any], None],
        model_name: str = "hey_jarvis",
        sensitivity: float = 0.5,
        vad_aggressiveness: int = 2,
        silence_ms: int = 900,
        max_record_s: float = 15.0,
        sample_rate: int = 16_000,
    ) -> None:
        # Lazy imports — package stays importable without wake-word extras.
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

        self._np = np
        self._sd = sd
        self._webrtcvad = webrtcvad
        self._oww_cls = _OWW

        self._model_name   = model_name
        self._sensitivity  = sensitivity
        self._vad_agg      = vad_aggressiveness
        self._silence_ms   = silence_ms
        self._max_record_s = max_record_s
        self._sample_rate  = sample_rate
        self._on_wake      = on_wake

        self._stop_event  = threading.Event()
        self._ptt_active  = threading.Event()   # set while PTT is held
        self._ptt_release = threading.Event()   # set when PTT is released
        self._thread: threading.Thread | None = None

        # Download pretrained models on first use (no-op if already cached).
        log.info("loading openwakeword model %r …", model_name)
        try:
            import openwakeword as _oww_pkg
            _oww_pkg.utils.download_models()
        except Exception:
            pass  # offline / already cached — model load below will catch real errors

        self._oww = _OWW(
            wakeword_models=[model_name],
            inference_framework="onnx",
        )
        log.info(
            "wake word listener ready — model=%r sensitivity=%.2f "
            "silence=%dms max=%.0fs",
            model_name, sensitivity, silence_ms, max_record_s,
        )

    # ---- public lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Spin up the background listener thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="friday-wakeword"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait for it to exit."""
        self._stop_event.set()
        # Unblock any pending PTT wait.
        self._ptt_release.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ---- PTT override (called from HotkeyController thread) ----------------

    def ptt_start(self) -> None:
        """Enter recording mode immediately — bypasses wake-word scoring."""
        self._ptt_release.clear()
        self._ptt_active.set()

    def ptt_stop(self) -> None:
        """End PTT recording and deliver the utterance via ``on_wake``."""
        self._ptt_active.clear()
        self._ptt_release.set()

    # ---- main listener loop ------------------------------------------------

    def _run(self) -> None:
        np = self._np
        sd = self._sd

        log.info(
            "always-on listener active — say %r or press PTT to talk",
            self._model_name,
        )
        vad = self._webrtcvad.Vad(self._vad_agg)

        try:
            # blocksize=0: we control read sizes manually, which lets us
            # use 1280-sample chunks for OWW and 480-sample frames for VAD
            # on the same open stream without reopening it.
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
            ) as stream:
                while not self._stop_event.is_set():
                    # ---- PTT override path --------------------------------
                    if self._ptt_active.is_set():
                        audio = self._record_ptt(stream)
                        self._deliver(audio)
                        continue

                    # ---- wake-word detection path -------------------------
                    try:
                        raw, _ = stream.read(_OWW_CHUNK_SAMPLES)
                    except Exception:
                        log.exception("mic read error in wake loop")
                        break

                    chunk = raw[:, 0]  # (N,1) → (N,)
                    score = self._score(chunk)

                    if score >= self._sensitivity:
                        log.info(
                            "wake word %r detected (score=%.3f) — recording",
                            self._model_name, score,
                        )
                        self._oww.reset()   # avoid double-fire on same phrase
                        audio = self._record_vad(stream, vad)
                        self._deliver(audio)

        except Exception:
            log.exception("wake word listener crashed")

    # ---- scoring -----------------------------------------------------------

    def _score(self, chunk_int16: Any) -> float:
        """Run openwakeword on one int16 chunk and return the max score."""
        np = self._np
        chunk_f32 = chunk_int16.astype(np.float32) / 32_768.0
        prediction: dict = self._oww.predict(chunk_f32)
        if not prediction:
            return 0.0
        # Key is the model name; use max() across all values in case the
        # name doesn't match exactly (e.g. custom model path vs basename).
        return float(max(prediction.values()))

    # ---- recording: VAD mode (after wake word) -----------------------------

    def _record_vad(self, stream: Any, vad: Any) -> Any:
        """Record until VAD detects silence or max_record_s elapses.

        Returns a float32 numpy array of the utterance.
        """
        np = self._np
        silence_frames_needed = max(1, self._silence_ms // _VAD_FRAME_MS)
        max_frames = int(self._max_record_s * 1_000 / _VAD_FRAME_MS)

        frames: list[Any] = []
        consecutive_silence = 0
        speech_started = False

        for _ in range(max_frames):
            if self._stop_event.is_set():
                break
            if self._ptt_active.is_set():
                # PTT pressed mid-wake-recording — let PTT path take over.
                break

            try:
                raw, _ = stream.read(_VAD_FRAME_SAMPLES)
            except Exception:
                log.exception("mic read error during VAD recording")
                break

            frame = raw[:, 0]
            frames.append(frame.copy())

            try:
                is_speech = vad.is_speech(frame.tobytes(), self._sample_rate)
            except Exception:
                is_speech = True   # on VAD error, assume speech

            if is_speech:
                speech_started = True
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            if speech_started and consecutive_silence >= silence_frames_needed:
                break

        if not frames:
            return np.zeros(0, dtype=np.float32)

        # Trim trailing silence (keep one frame so the cut sounds natural).
        keep = max(1, len(frames) - silence_frames_needed + 1)
        frames = frames[:keep]

        audio_i16  = np.concatenate(frames)
        audio_f32  = audio_i16.astype(np.float32) / 32_768.0
        duration   = len(audio_f32) / self._sample_rate
        log.info("utterance recorded: %.2fs (VAD mode)", duration)
        return audio_f32

    # ---- recording: PTT mode -----------------------------------------------

    def _record_ptt(self, stream: Any) -> Any:
        """Record until PTT is released or max_record_s elapses."""
        np = self._np
        max_samples = int(self._max_record_s * self._sample_rate)
        frames: list[Any] = []
        total_samples = 0

        while (
            not self._stop_event.is_set()
            and not self._ptt_release.is_set()
            and total_samples < max_samples
        ):
            try:
                raw, _ = stream.read(_VAD_FRAME_SAMPLES)
            except Exception:
                log.exception("mic read error during PTT recording")
                break
            frame = raw[:, 0]
            frames.append(frame.copy())
            total_samples += len(frame)

        # Drain the release event so it doesn't persist.
        self._ptt_release.clear()

        if not frames:
            return np.zeros(0, dtype=np.float32)

        audio_i16 = np.concatenate(frames)
        audio_f32 = audio_i16.astype(np.float32) / 32_768.0
        duration  = len(audio_f32) / self._sample_rate
        log.info("utterance recorded: %.2fs (PTT mode)", duration)
        return audio_f32

    # ---- delivery ----------------------------------------------------------

    def _deliver(self, audio: Any) -> None:
        """Hand a completed utterance to the on_wake callback."""
        if audio is None or len(audio) == 0:
            return
        try:
            self._on_wake(audio)
        except Exception:
            log.exception("on_wake callback raised")


__all__ = ["WakeWordListener"]
