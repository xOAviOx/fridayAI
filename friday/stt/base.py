"""STT provider interface.

Phase 0: definition only. Phase 1 lands ``groq_whisper.GroqWhisperSTT``
and ``local_whisper.LocalWhisperSTT`` can be dropped in later without
touching the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    """One transcription result.

    ``text`` is the only required field. Providers that can report
    language / confidence / latency populate the optional fields so the
    agent loop can show them in the HUD later.
    """

    text: str
    language: str | None = None
    duration_s: float | None = None
    latency_ms: int | None = None


class STTProvider(ABC):
    """Turn a PCM-audio blob into text.

    Implementations must be safe to call from the main thread (the MVP
    loop is synchronous). Streaming STT is added in Phase 2 by extending
    this interface, not by replacing it.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> Transcript:
        """Transcribe ``audio`` (16-bit signed PCM) to text."""

    def close(self) -> None:  # pragma: no cover - default is no-op
        """Release any underlying resources. Default is a no-op."""
