"""ElevenLabs cloud TTS provider.

Streams audio from ElevenLabs' ``/v1/text-to-speech/{voice_id}/stream``
endpoint as 16-bit signed little-endian PCM at 24 kHz — the same shape
:class:`~friday.tts.kokoro.KokoroTTS` produces, so the playback layer
doesn't need to re-detect format mid-conversation.

Why streaming HTTP and not the websocket?
    The websocket variant is only worth its complexity when text is
    arriving token-by-token from the LLM. FRIDAY already chunks LLM
    output into whole sentences via :class:`SentenceSplitter` before
    handing them to TTS, so per-sentence HTTP streaming gives the same
    perceived latency with a much simpler failure model.

Why PCM and not MP3?
    The playback layer (``audio/playback.py``) expects raw PCM in
    ``AudioChunk``. Asking ElevenLabs for ``pcm_24000`` skips a decode
    step entirely.

Barge-in
    ``stop()`` closes the live HTTP response, which unblocks the
    streaming iterator on the next read. The agent's existing TTS
    pipeline already calls ``stop()`` when the user starts talking.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator

from friday.tts.base import AudioChunk, TTSProvider

log = logging.getLogger(__name__)

# Match KokoroTTS so the speaker stays at a single rate across providers.
_SAMPLE_RATE = 24_000
# 2 bytes per sample @ 24 kHz mono = 48 kB/s. ~6 kB per read ≈ 125 ms
# of audio per chunk — small enough to start playback fast, large
# enough to avoid syscall overhead on the speaker side.
_READ_BYTES = 6_144
# ElevenLabs default endpoint. Override only for a regional mirror.
_DEFAULT_BASE_URL = "https://api.elevenlabs.io"


class ElevenLabsTTS(TTSProvider):
    """Cloud TTS via ElevenLabs streaming HTTP.

    Parameters
    ----------
    api_key:
        ElevenLabs API key (from .env as ``ELEVENLABS_API_KEY``).
    voice_id:
        Voice id from your ElevenLabs voice library. Required — there
        is no sensible default; the free starter voices each have their
        own id.
    model:
        ElevenLabs model id. ``eleven_turbo_v2_5`` is the lowest-latency
        general-purpose model and the project default.
    base_url:
        Override only if you're pointing at a custom proxy.
    timeout_s:
        HTTP timeout for the *initial* response. Streaming reads don't
        share this budget — a long utterance won't time out just
        because synthesis takes a while.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2_5",
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabsTTS requires an api_key")
        if not voice_id:
            raise ValueError(
                "ElevenLabsTTS requires a voice_id — set tts.elevenlabs.voice_id "
                "in config.yaml (find ids at https://elevenlabs.io/app/voice-library)"
            )

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "ElevenLabsTTS requires httpx. Install with: "
                "pip install -e '.[tts-elevenlabs]'"
            ) from exc

        self._httpx = httpx
        self._client: Any = httpx.Client(
            base_url=base_url,
            headers={
                "xi-api-key": api_key,
                "accept": "audio/pcm",
                "content-type": "application/json",
            },
            timeout=timeout_s,
        )
        self._voice_id = voice_id
        self._model = model

        # In-flight response, held so stop() can close it from another
        # thread (the barge-in listener).  Guarded by a lock because the
        # speaker thread races with the keypress thread on stop().
        self._lock = threading.Lock()
        self._active: Any = None
        self._stopped = threading.Event()

        log.info(
            "ElevenLabsTTS ready: voice=%s model=%s rate=%dHz",
            voice_id, model, _SAMPLE_RATE,
        )

    # ---- TTSProvider API ----------------------------------------------------

    def synthesize(self, text: str) -> Iterator[AudioChunk]:
        text = (text or "").strip()
        if not text:
            return

        # Clear any previous stop signal — a fresh utterance starts unblocked.
        self._stopped.clear()

        url = f"/v1/text-to-speech/{self._voice_id}/stream"
        params = {"output_format": f"pcm_{_SAMPLE_RATE}"}
        body = {
            "text": text,
            "model_id": self._model,
        }

        try:
            response_cm = self._client.stream("POST", url, params=params, json=body)
        except Exception as exc:  # pragma: no cover - network error path
            log.error("elevenlabs request failed to send: %s", exc)
            return

        # `stream(...)` returns a context manager — we enter it manually
        # so stop() can close it from another thread.  __exit__ runs in
        # the finally block.
        response = response_cm.__enter__()
        try:
            with self._lock:
                self._active = response

            try:
                response.raise_for_status()
            except self._httpx.HTTPStatusError as exc:
                # Drain whatever body came back so the error message is
                # human-readable instead of "<stream>".
                try:
                    detail = b"".join(response.iter_bytes()).decode("utf-8", "replace")
                except Exception:
                    detail = ""
                log.error(
                    "elevenlabs %s: %s",
                    exc.response.status_code, detail[:300] or "<no body>",
                )
                return

            for chunk in response.iter_bytes(chunk_size=_READ_BYTES):
                if self._stopped.is_set():
                    break
                if not chunk:
                    continue
                yield AudioChunk(
                    pcm=chunk, sample_rate=_SAMPLE_RATE, channels=1
                )
        finally:
            with self._lock:
                self._active = None
            try:
                response_cm.__exit__(None, None, None)
            except Exception:  # pragma: no cover - close-during-abort
                pass

    def stop(self) -> None:
        """Abort the in-flight stream (barge-in)."""
        self._stopped.set()
        with self._lock:
            response = self._active
        if response is not None:
            try:
                response.close()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        self.stop()
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass


__all__ = ["ElevenLabsTTS"]
