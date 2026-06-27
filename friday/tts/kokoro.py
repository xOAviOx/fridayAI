"""Kokoro local TTS provider.

`Kokoro <https://github.com/hexgrad/kokoro>`_ is an ~82M-parameter open-weight
TTS model that runs CPU-only in roughly a second per sentence on a modern
laptop. No API key, no network round-trip, no per-character billing — which
makes it the right default for FRIDAY's zero-cost stack.

Kokoro emits 24 kHz mono float32 samples. This wrapper converts each
internal segment to 16-bit signed little-endian PCM (the format the
:class:`~friday.tts.base.AudioChunk` contract promises) and yields one
chunk per segment, so the playback layer can start speaking before the
full utterance is finished synthesising. That's also what keeps the
Phase 2 barge-in story honest: ``stop()`` short-circuits the iterator
between segments instead of waiting for the whole sentence to render.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator

from friday.tts.base import AudioChunk, TTSProvider

log = logging.getLogger(__name__)

# Kokoro is fixed at 24 kHz mono in the upstream model — not a config knob.
_KOKORO_SAMPLE_RATE = 24_000

# Single-character language codes accepted by KPipeline. Kept here so a
# typo in config.yaml dies with a friendly error instead of a torch crash
# halfway through model load.
_VALID_LANG_CODES = frozenset("abefhijpz")


class KokoroTTS(TTSProvider):
    """Local Kokoro TTS — no API key, no network, no cost.

    Parameters
    ----------
    voice:
        Kokoro voice id (e.g. ``af_heart``, ``af_bella``, ``am_adam``,
        ``bf_emma``). The full list lives on the model card; ``af_heart``
        is the upstream default and a good neutral American English voice.
    lang_code:
        Single-character Kokoro language code. ``"a"`` is American English
        (the default and what matches the ``a*`` voice family), ``"b"`` is
        British English, and a handful of others exist for non-English.
    speed:
        Playback speed multiplier. ``1.0`` is the model's natural rate;
        ``1.1``–``1.2`` is a common "snappier assistant" setting.
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
    ) -> None:
        if lang_code not in _VALID_LANG_CODES:
            raise ValueError(
                f"tts.kokoro.lang_code must be one of {sorted(_VALID_LANG_CODES)!r}, "
                f"got {lang_code!r}"
            )
        if speed <= 0:
            raise ValueError(f"tts.kokoro.speed must be > 0, got {speed!r}")

        # Lazy import so Phase 0 `python -m friday.main` doesn't need
        # the local-TTS extras installed just to validate config.
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Kokoro TTS selected but the `kokoro` package isn't installed. "
                "Install the local-TTS extras:\n"
                "    uv sync --extra tts-kokoro\n"
                "or:\n"
                "    pip install -e '.[tts-kokoro]'"
            ) from exc

        self._pipeline = KPipeline(lang_code=lang_code)
        self._voice = voice
        self._speed = speed
        # Set by ``stop()`` to break out of the segment loop on barge-in.
        # Cleared at the start of every ``synthesize`` call so a fresh
        # utterance isn't pre-cancelled.
        self._stop_flag = threading.Event()
        log.debug(
            "Kokoro ready: voice=%s lang_code=%s speed=%s sr=%d",
            voice,
            lang_code,
            speed,
            _KOKORO_SAMPLE_RATE,
        )

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio chunks this provider emits."""
        return _KOKORO_SAMPLE_RATE

    def synthesize(self, text: str) -> Iterator[AudioChunk]:
        """Yield 16-bit PCM chunks, one per Kokoro segment.

        Kokoro segments long input by sentence internally, so the first
        chunk typically lands within ~1 second on CPU — fast enough that
        the playback layer can start before the rest of the utterance is
        done. Empty / whitespace-only text yields nothing.
        """
        if not text or not text.strip():
            return

        self._stop_flag.clear()

        generator = self._pipeline(text, voice=self._voice, speed=self._speed)
        for _graphemes, _phonemes, audio in generator:
            if self._stop_flag.is_set():
                log.debug("Kokoro synthesis stopped early (barge-in)")
                return
            yield AudioChunk(
                pcm=_to_pcm16_bytes(audio),
                sample_rate=_KOKORO_SAMPLE_RATE,
                channels=1,
            )

    def stop(self) -> None:
        """Signal the synthesize loop to stop emitting further chunks.

        Kokoro has no per-frame cancel API — the current segment runs to
        completion in the background — but we skip emitting any further
        segments, which is what bounds barge-in latency to one sentence
        of overshoot at worst. The playback layer is what actually drops
        already-yielded audio when the user starts talking.
        """
        self._stop_flag.set()

    def close(self) -> None:
        """Drop the pipeline reference so the model can be GC'd."""
        self._pipeline = None


def _to_pcm16_bytes(audio: Any) -> bytes:
    """Convert Kokoro's float32 mono samples to 16-bit signed LE PCM bytes.

    Kokoro returns either a ``torch.Tensor`` or a numpy array depending on
    version. We avoid importing torch eagerly (it's already loaded via
    Kokoro at this point, but we don't want a hard dependency in the
    annotations) and just feature-detect ``.detach``.
    """
    import numpy as np  # local import keeps base TTS import torch-free

    if hasattr(audio, "detach"):  # torch.Tensor
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    # Clip to avoid wraparound when the model overshoots [-1.0, 1.0].
    np.clip(arr, -1.0, 1.0, out=arr)
    pcm = (arr * 32767.0).astype(np.int16, copy=False)
    return pcm.tobytes()


__all__ = ["KokoroTTS"]
