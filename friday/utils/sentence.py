"""Sentence-level text buffer for streaming LLM → TTS.

Accumulates token deltas from a streaming LLM and yields complete
sentences so the TTS layer can start speaking before the model has
finished generating the full reply.

Usage::

    splitter = SentenceSplitter()
    for token in llm_stream:
        for sentence in splitter.push(token):
            tts.speak(sentence)
    for sentence in splitter.flush():
        tts.speak(sentence)  # trailing fragment

Design notes
------------
* Splits on ``.``, ``!``, ``?``, ``…`` followed by whitespace *or*
  the end of the buffer.
* Skips fragments shorter than ``min_length`` (default 10 chars)
  so "Dr. Smith" or "Mr. X" don't fire a TTS call for "Dr." alone.
* State is internal — one instance per LLM turn; reset between turns
  by constructing a new one or calling ``reset()``.
"""

from __future__ import annotations

import re

# Terminal punctuation that ends a spoken sentence. The positive
# lookbehind means the character itself is kept; the \s+ consumes the
# whitespace separator so the *next* sentence doesn't start with it.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

# Minimum buffer length before we bother splitting. This avoids
# emitting "Dr." or "U.S." as standalone sentences before enough
# context is accumulated. 10 chars is conservative on purpose.
_DEFAULT_MIN = 10


class SentenceSplitter:
    """Accumulate streaming tokens and yield complete sentences.

    Parameters
    ----------
    min_length:
        Minimum character count a candidate sentence must reach
        before it's yielded. Prevents spurious splits on
        abbreviations such as ``"Dr."``.
    """

    def __init__(self, *, min_length: int = _DEFAULT_MIN) -> None:
        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        self._min = min_length
        self._buf = ""

    # ------------------------------------------------------------------ #

    def push(self, token: str) -> list[str]:
        """Append *token* to the buffer and return any complete sentences.

        Returns an empty list when the buffer doesn't yet contain a
        complete sentence.
        """
        if not token:
            return []
        self._buf += token
        return self._drain()

    def flush(self) -> list[str]:
        """Return the remaining buffer as a sentence (if non-empty).

        Call this once the stream is exhausted to emit any trailing
        text that doesn't end with terminal punctuation.
        """
        text = self._buf.strip()
        self._buf = ""
        return [text] if text else []

    def reset(self) -> None:
        """Clear the buffer — reuse the instance for a new turn."""
        self._buf = ""

    # ------------------------------------------------------------------ #

    def _drain(self) -> list[str]:
        """Split the buffer on sentence boundaries and return complete ones."""
        sentences: list[str] = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if m is None:
                break
            candidate = self._buf[: m.start() + 1].strip()  # include punct
            rest = self._buf[m.end() :]

            if len(candidate) < self._min:
                # Too short — could be an abbreviation. Keep scanning
                # from the character after the boundary match so we
                # don't loop forever on the same match.
                if not rest:
                    break
                self._buf = rest
                continue

            sentences.append(candidate)
            self._buf = rest

        return sentences


__all__ = ["SentenceSplitter"]
