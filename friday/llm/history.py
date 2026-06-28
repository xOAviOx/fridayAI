"""Sliding-window conversation history for the agent loop.

Free-tier TPM is the bottleneck Phase 1 lives under, and the cheapest
way to keep it under control is to never send the model more turns
than it actually needs to follow the conversation. :class:`SlidingWindowHistory`
keeps the system prompt sticky and trims older user/assistant
exchanges off the front as new ones arrive.

A "turn" here is anchored on user messages — that's the only role the
window count looks at. Assistant replies (including tool-call ones)
and tool-result messages tag along with whichever user turn they
belong to, so a tool-call/tool-result pair is never split.
"""

from __future__ import annotations

import logging

from friday.llm.base import ChatMessage

log = logging.getLogger(__name__)


class SlidingWindowHistory:
    """Bounded conversation history with a sticky system message.

    Parameters
    ----------
    system:
        Optional system message. Stored as the first element of every
        :meth:`messages` result; never trimmed.
    max_user_turns:
        How many of the most recent user turns to keep. The default of
        8 matches the brief's "lean prompt + sliding window" guidance
        for the Groq free tier without truncating the average request.
    """

    def __init__(
        self,
        *,
        system: ChatMessage | str | None = None,
        max_user_turns: int = 8,
    ) -> None:
        if max_user_turns <= 0:
            raise ValueError(f"max_user_turns must be > 0, got {max_user_turns}")
        if isinstance(system, str):
            system = ChatMessage(role="system", content=system)
        if system is not None and system.role != "system":
            raise ValueError(
                f"system message must have role='system', got {system.role!r}"
            )
        self._system = system
        self._max_user_turns = max_user_turns
        self._history: list[ChatMessage] = []

    def add(self, message: ChatMessage) -> None:
        """Append a message to the history.

        System messages passed here are rejected — set the system
        message at construction time so the window invariant is clear.
        """
        if message.role == "system":
            raise ValueError(
                "system message can't be added through add(); pass it to the "
                "constructor instead"
            )
        self._history.append(message)

    def extend(self, messages: list[ChatMessage]) -> None:
        for m in messages:
            self.add(m)

    def messages(self) -> list[ChatMessage]:
        """Return the current windowed view as a flat list.

        The first element (if any) is the sticky system message; the
        rest is the tail of ``_history`` starting at the
        ``max_user_turns``-th most-recent user message. Anything between
        user messages (assistant replies, tool-call / tool-result
        exchanges) is preserved verbatim.
        """
        windowed: list[ChatMessage] = []
        if self._system is not None:
            windowed.append(self._system)

        user_indices = [
            i for i, m in enumerate(self._history) if m.role == "user"
        ]
        if len(user_indices) <= self._max_user_turns:
            windowed.extend(self._history)
        else:
            start = user_indices[-self._max_user_turns]
            windowed.extend(self._history[start:])
        return windowed

    def reset(self) -> None:
        """Drop all non-system history. Use between independent sessions."""
        self._history.clear()

    def __len__(self) -> int:
        """Length of the *windowed* view, including the system message."""
        return len(self.messages())


__all__ = ["SlidingWindowHistory"]
