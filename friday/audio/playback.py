"""TTS playback via sounddevice.

``Speaker.play(chunks)`` drains an iterator of
:class:`~friday.tts.base.AudioChunk` objects to the default output
device, opening an ``sd.OutputStream`` lazily on the first chunk so the
sample rate / channel count can come from the audio itself rather than
being hardwired (Kokoro is 24 kHz; a future provider might not be).

Phase 1 is synchronous: ``play()`` blocks until the iterator is
exhausted (or ``stop()`` cuts it short). The streaming/barge-in story
in Phase 2 will keep the same API — ``stop()`` already supports a hard
cut via ``stream.abort()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from friday.tts.base import AudioChunk

log = logging.getLogger(__name__)


class Speaker:
    """Plays a stream of ``AudioChunk`` objects through the default device.

    The stream is configured from the first chunk and held open for the
    duration of the iterator. Mid-utterance ``stop()`` calls
    ``stream.abort()``, which drops pending samples immediately — the
    contract Phase 2 barge-in needs.
    """

    def __init__(self) -> None:
        # Lazy import — keeps the module importable without the extras.
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Playback requires sounddevice + numpy. Install the audio extras:\n"
                "    uv sync --extra audio\n"
                "or:\n"
                "    pip install -e '.[audio]'"
            ) from exc

        self._sd = sd
        self._np = np
        self._stop_flag = threading.Event()
        # Held while a stream exists, so ``stop()`` from another thread
        # can call ``abort()`` safely without racing teardown.
        self._stream_lock = threading.Lock()
        self._stream: Any = None

    def play(self, chunks: Iterable[AudioChunk]) -> None:
        """Drain ``chunks`` to the speaker. Blocks until done or stopped.

        Empty iterators are a no-op. Sample rate / channel count are
        taken from the first chunk; subsequent chunks must match
        (changing rate mid-utterance would require reopening the
        stream and isn't a real-world case here).
        """
        self._stop_flag.clear()
        iterator = iter(chunks)

        try:
            first = next(iterator)
        except StopIteration:
            return

        sample_rate = first.sample_rate
        channels = first.channels

        with self._sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        ) as stream:
            with self._stream_lock:
                self._stream = stream

            try:
                self._write_chunk(stream, first, sample_rate, channels)
                for chunk in iterator:
                    if self._stop_flag.is_set():
                        log.debug("playback stopped mid-stream")
                        return
                    self._write_chunk(stream, chunk, sample_rate, channels)
            finally:
                with self._stream_lock:
                    self._stream = None

    def stop(self) -> None:
        """Cut playback immediately. Safe to call from any thread.

        Sets the stop flag (so the next chunk write is skipped) and
        aborts the live stream (so audio already queued to the device
        is dropped). One or both will be a no-op depending on timing —
        we just need at least one to fire.
        """
        self._stop_flag.set()
        with self._stream_lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # pragma: no cover - defensive
                log.exception("OutputStream.abort raised")

    # ----- internal -----------------------------------------------------

    def _write_chunk(
        self,
        stream: Any,
        chunk: AudioChunk,
        expected_rate: int,
        expected_channels: int,
    ) -> None:
        if chunk.sample_rate != expected_rate:
            raise ValueError(
                f"AudioChunk sample_rate changed mid-stream "
                f"({chunk.sample_rate} != {expected_rate})"
            )
        if chunk.channels != expected_channels:
            raise ValueError(
                f"AudioChunk channels changed mid-stream "
                f"({chunk.channels} != {expected_channels})"
            )
        arr = self._np.frombuffer(chunk.pcm, dtype=self._np.int16)
        if expected_channels > 1:
            arr = arr.reshape(-1, expected_channels)
        stream.write(arr)


__all__ = ["Speaker"]
