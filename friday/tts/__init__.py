"""Text-to-speech providers (the "mouth")."""

from friday.tts.base import AudioChunk, TTSProvider
from friday.tts.kokoro import KokoroTTS

__all__ = ["AudioChunk", "KokoroTTS", "TTSProvider"]
