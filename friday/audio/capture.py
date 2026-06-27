"""Push-to-talk microphone capture.

The recorder is intentionally dumb: ``start()`` opens an
``sd.InputStream`` that appends every callback buffer to an in-memory
list; ``stop()`` closes the stream and returns the concatenated buffer
as a single float32 mono array. STT-shaped concerns (WAV wrapping, VAD,
silence trim) live one layer up — this file just gets audio off the
mic.

Audio is captured as float32 at the configured sample rate (default
16 kHz mono — matches Whisper's expected input rate, so no resampling
is needed before STT in chunk 2).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


class MicRecorder:
    """Microphone recorder driven by an external start/stop signal.

    Parameters
    ----------
    sample_rate:
        Capture rate in Hz. 16 000 matches Whisper's native input and
        avoids a resample step in chunk 2.
    channels:
        1 (mono) for STT. Stereo is not useful here and just doubles
        the buffer size we have to ship to Groq.
    blocksize:
        How many frames per callback. 1024 is a good default — ~64 ms
        of latency at 16 kHz, well below human perception of "lag".
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        blocksize: int = 1024,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {channels}")

        # Lazy import — leaves friday.audio importable without the extras.
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Mic capture requires sounddevice + numpy. Install the audio extras:\n"
                "    uv sync --extra audio\n"
                "or:\n"
                "    pip install -e '.[audio]'"
            ) from exc

        self._sd = sd
        self._np = np
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize

        self._stream: Any = None
        self._chunks: list[Any] = []  # list[np.ndarray]
        # Guards _chunks against the audio callback racing stop().
        self._lock = threading.Lock()
        self._recording = False

    def start(self) -> None:
        """Open the input stream and begin appending audio to the buffer.

        Calling ``start()`` while already recording is a no-op so a
        stuck-key event doesn't reset the buffer mid-utterance.
        """
        if self._recording:
            return
        self._chunks = []
        self._recording = True
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        log.debug("recording started: sr=%d ch=%d", self.sample_rate, self.channels)

    def stop(self) -> "Any":
        """Close the stream and return the captured audio.

        Returns
        -------
        numpy.ndarray
            Mono float32 array of length ``frames`` (or shape
            ``(frames, 2)`` if ``channels == 2``). Empty array if
            ``stop()`` is called without a matching ``start()``.
        """
        if not self._recording:
            return self._np.zeros(0, dtype=self._np.float32)
        self._recording = False
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        with self._lock:
            chunks = self._chunks
            self._chunks = []
        if not chunks:
            return self._np.zeros(0, dtype=self._np.float32)
        audio = self._np.concatenate(chunks, axis=0)
        # Mono path: squeeze the singleton channel axis so downstream
        # code doesn't have to special-case shape (N, 1) vs (N,).
        if self.channels == 1 and audio.ndim == 2:
            audio = audio[:, 0]
        log.debug(
            "recording stopped: %d frames (%.2fs)",
            len(audio),
            len(audio) / self.sample_rate,
        )
        return audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ----- internal -----------------------------------------------------

    def _callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        # sounddevice runs this on the PortAudio thread. Keep it quick.
        if status:
            # XRuns happen if the system can't keep up. Log but keep going.
            log.debug("input stream status: %s", status)
        with self._lock:
            # Copy because PortAudio reuses the buffer after callback returns.
            self._chunks.append(indata.copy())


__all__ = ["MicRecorder"]
