"""Groq-hosted Whisper STT provider.

Groq runs Whisper at substantially higher throughput than OpenAI's own
endpoint and is free up to the rate limits documented in
``config.yaml``. The endpoint is OpenAI-SDK-compatible, so we drive it
through the ``openai`` Python client with ``base_url`` swapped to
``https://api.groq.com/openai/v1`` — same wire protocol, different host.

What this module does
---------------------
1. Accept the 16-bit signed PCM bytes the :class:`STTProvider` interface
   promises (callers convert their mic buffer with
   :func:`friday.audio.encoding.float32_to_pcm16_bytes`).
2. Wrap those raw samples in a minimal WAV container *in memory* — no
   disk I/O. Whisper rejects naked PCM but happily eats WAV, and the
   stdlib ``wave`` module is enough to build the header.
3. POST it as multipart/form-data via ``openai.audio.transcriptions``.
4. Honor ``retry-after`` on 429s with a small bounded retry. Full
   request/token-budget tracking lands with chunk 5 (`utils/tokens.py`);
   we just don't want a single rate-limit hiccup to surface as a stack
   trace mid-conversation.

Phase 1 is synchronous: one HTTP round-trip per utterance, blocking the
caller until the transcript comes back. Streaming STT is a Phase 2
problem and would extend this class rather than replace it.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any

from friday.stt.base import STTProvider, Transcript

log = logging.getLogger(__name__)

# Generous enough for a multi-minute utterance on a slow uplink, but
# short enough that a stuck connection surfaces fast.
_DEFAULT_TIMEOUT_S = 30.0

# Bounded 429 retries. Groq's free tier is bursty enough that one
# automatic retry usually clears it; we cap at three to avoid
# pathological waits during an outage.
_MAX_RETRIES = 3


class GroqWhisperSTT(STTProvider):
    """Groq Whisper STT.

    Parameters
    ----------
    api_key:
        Groq API key. Loaded by ``friday.config`` from ``GROQ_API_KEY``
        in ``.env``; the caller passes ``config.secrets.groq_api_key``.
    model:
        Groq Whisper model id. ``whisper-large-v3-turbo`` is the right
        default — full Whisper-large quality at ~10× the throughput.
    base_url:
        OpenAI-SDK-compatible base URL. Default is Groq's; swappable
        only because the rest of the code paths are model-agnostic.
    timeout_s:
        Per-request HTTP timeout. The default is 30s — plenty for a
        single utterance over a normal connection.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GroqWhisperSTT requires a non-empty api_key")

        # Lazy import so Phase 0 `python -m friday.main` doesn't need
        # the STT extras installed just to validate config.
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Groq Whisper STT selected but the `openai` package isn't installed. "
                "Install the STT extras:\n"
                "    uv sync --extra stt-groq\n"
                "or:\n"
                "    pip install -e '.[stt-groq]'"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )
        self._model = model
        log.debug(
            "GroqWhisperSTT ready: model=%s base_url=%s timeout=%.1fs",
            model,
            base_url,
            timeout_s,
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> Transcript:
        """Transcribe a PCM buffer to text.

        ``audio`` must be 16-bit signed little-endian PCM (mono). Empty
        input returns an empty :class:`Transcript` immediately — no
        request is made, which keeps the free-tier RPM budget clean
        during silence.
        """
        if not audio:
            return Transcript(text="", language=language)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        wav_bytes = _wrap_pcm16_as_wav(audio, sample_rate=sample_rate)
        # 2 bytes per sample (16-bit) * 1 channel — divide once.
        duration_s = len(audio) / (2 * sample_rate)

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            # OpenAI multipart contract: (filename, bytes, content_type).
            # The filename has to look like a real audio file; "audio.wav"
            # is enough for the server to pick the right decoder.
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        if language:
            request_kwargs["language"] = language

        start = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.audio.transcriptions.create(**request_kwargs)
            except Exception as exc:  # noqa: BLE001 — SDK error hierarchy varies
                last_exc = exc
                retry_after = _retry_after_seconds(exc)
                if retry_after is None or attempt == _MAX_RETRIES:
                    raise
                log.warning(
                    "Groq Whisper rate-limited (429); sleeping %.1fs "
                    "before retry %d/%d",
                    retry_after,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(retry_after)
                continue

            latency_ms = int((time.monotonic() - start) * 1000)
            text = (getattr(response, "text", "") or "").strip()
            log.debug(
                "Groq Whisper: %.2fs audio → %d chars in %d ms",
                duration_s,
                len(text),
                latency_ms,
            )
            return Transcript(
                text=text,
                language=language,
                duration_s=duration_s,
                latency_ms=latency_ms,
            )

        # Unreachable — the loop above either returns or raises — but
        # mypy can't see that.
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        """Close the underlying HTTP client. Safe to call multiple times."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort
            pass
        self._client = None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _wrap_pcm16_as_wav(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    """Wrap raw 16-bit signed PCM in a WAV container, in memory.

    The stdlib ``wave`` module writes a valid RIFF/WAVE header for us
    in a couple of microseconds — no temp file, no extra dependency.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return the ``retry-after`` value in seconds for a 429, else ``None``.

    The openai SDK's exception layout has shifted across majors, so we
    dig defensively. If the exception isn't a rate-limit error, or the
    server didn't include a numeric ``retry-after``, we return ``None``
    and the caller re-raises.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    if status != 429:
        return None
    headers = getattr(response, "headers", None) or {}
    # Both Groq and OpenAI send seconds as a number (sometimes as a
    # decimal string). HTTP-date is allowed by the spec but rare here;
    # ignoring it just means we re-raise instead of looping.
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


__all__ = ["GroqWhisperSTT"]
