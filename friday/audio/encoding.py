"""Tiny audio-format helpers shared between capture, STT, and playback.

Kept dependency-free at module-import time — numpy is imported lazily
inside the helpers so ``friday.audio.encoding`` itself stays importable
without the audio extras installed. That matters because the STT layer
needs this conversion (mic float32 → 16-bit PCM bytes for the wire) and
the audio extras are otherwise optional.
"""

from __future__ import annotations

from typing import Any


def float32_to_pcm16_bytes(audio: Any) -> bytes:
    """Convert a float32 mono PCM array to 16-bit signed LE PCM bytes.

    ``MicRecorder.stop()`` returns float32 samples in ``[-1.0, 1.0]``;
    the :class:`~friday.stt.base.STTProvider` interface (and every wire
    format we care about) expects 16-bit signed PCM. This is the
    cheapest correct path: clip to range, scale, cast.

    Multi-channel input is downmixed to mono — STT gets nothing from a
    second channel and Groq just charges us for the extra bytes.
    """
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        # (frames, channels) → (frames,). MicRecorder pre-squeezes mono
        # already, but be defensive for any stereo caller upstream.
        arr = arr.mean(axis=-1, dtype=np.float32)
    # Clip in place — Kokoro can overshoot ±1.0 slightly and the cast
    # below would wrap if we didn't.
    np.clip(arr, -1.0, 1.0, out=arr)
    pcm = (arr * 32767.0).astype(np.int16, copy=False)
    return pcm.tobytes()


__all__ = ["float32_to_pcm16_bytes"]
