"""Route the LLM's reply to the next action.

In Phase 1 the router has one job: read a :class:`ChatResponse` and
produce a :class:`RouteResult` listing the speech to play and the
tool calls to dispatch. The agent loop reads that result and decides
what to do next; the executor handles the tool calls one at a time.

Phase 3+ this is where the three-tier routing (skill registry → code
exec → computer use) lives, plus the Anthropic escalation path for
hard reasoning. For now the registry is the only enabled tier, so the
router just forwards whatever the model picked.

The router is a pure function (no I/O), which is exactly what the spec
calls out as needing tests — see ``tests/test_router.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from friday.llm.base import ChatResponse, ToolCall

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    """What the loop should do after the model's latest reply.

    Attributes
    ----------
    speech:
        Text to synthesise and play through the speakers. Empty when
        the model only emitted tool calls.
    tool_calls:
        Tools to dispatch (each through the safety gate + executor),
        in the order the model returned them. Empty when the model
        only emitted prose.
    """

    speech: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return bool(self.speech and self.speech.strip())

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_terminal(self) -> bool:
        """True when the model is done with this turn.

        The loop stops feeding the model back into itself once there
        are no more tool calls to make — at that point the turn is
        finished and we wait for the next user utterance.
        """
        return not self.has_tool_calls


class Router:
    """Decide what the loop does after the model speaks.

    Stateless. One instance per agent loop is fine; constructing a
    fresh one per turn would work too.
    """

    def route(self, response: ChatResponse) -> RouteResult:
        """Split a ``ChatResponse`` into (speech, tool_calls)."""
        speech = (response.content or "").strip()
        tool_calls = list(response.tool_calls)
        log.debug(
            "router: speech=%d chars, tool_calls=%s, finish=%s",
            len(speech),
            [tc.name for tc in tool_calls],
            response.finish_reason,
        )
        return RouteResult(speech=speech, tool_calls=tool_calls)


__all__ = ["RouteResult", "Router"]
