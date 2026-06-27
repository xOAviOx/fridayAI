"""TTS provider interface.

The interface is iterator-shaped from day one so that Phase 2's streaming
ElevenLabs implementation (starts speaking before the LLM has finished
generating) doesn't need an API change. A non-streaming provider can
simply yield a single chunk.

Phase 0: definition only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class AudioChunk:
    """A single piece of synthesized PCM audio.

    ``pcm`` is raw 16-bit signed little-endian samples — playback in
    ``audio/playback.py`` (Phase 1) is responsible for shipping it to
    the sound device. Sample rate / channel count are reported here so
    the player doesn't have to be reconfigured per provider.
    """

    pcm: bytes
    sample_rate: int
    channels: int = 1


class TTSProvider(ABC):
    """Synthesize speech from text.

    The agent uses ``synthesize`` to get an iterable of chunks; the
    playback layer (not this interface) is what gets interrupted on
    barge-in. ``stop`` is a hint for providers that hold long-lived
    connections (ElevenLabs websocket) — most implementations can
    leave it as a no-op.
    """

    @abstractmethod
    def synthesize(self, text: str) -> Iterator[AudioChunk]:
        """Yield audio chunks for ``text``. Blocks until the next chunk."""

    def stop(self) -> None:  # pragma: no cover - default is no-op
        """Hint that the consumer is done early (barge-in / panic)."""

    def close(self) -> None:  # pragma: no cover - default is no-op
        """Release any underlying resources. Default is a no-op."""
