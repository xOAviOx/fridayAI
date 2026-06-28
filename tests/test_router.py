"""Tests for ``friday.agent.router.Router``.

The brief calls these out by name: the router and the @skill schema
generator are the two components Phase 1 specifically requires tests
for. The router is a pure function (ChatResponse → RouteResult), so
each test just feeds a synthetic response and asserts the split.
"""

from __future__ import annotations

from friday.agent.router import RouteResult, Router
from friday.llm.base import ChatResponse, ToolCall


def _r(
    *,
    content: str | None = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )


# --------------------------------------------------------------------------- #
# Pure-text replies                                                           #
# --------------------------------------------------------------------------- #


def test_plain_text_reply_routes_to_speech_only() -> None:
    router = Router()
    out = router.route(_r(content="Hello there."))
    assert out.speech == "Hello there."
    assert out.tool_calls == []
    assert out.has_speech
    assert not out.has_tool_calls
    assert out.is_terminal


def test_empty_content_yields_empty_speech() -> None:
    router = Router()
    out = router.route(_r(content=None))
    assert out.speech == ""
    assert not out.has_speech
    assert out.is_terminal


def test_whitespace_only_content_does_not_count_as_speech() -> None:
    out = Router().route(_r(content="   \n   "))
    assert not out.has_speech
    # The result still carries the original content stripped to "",
    # so the loop won't try to synthesise blank audio.
    assert out.speech == ""


# --------------------------------------------------------------------------- #
# Tool-call replies                                                           #
# --------------------------------------------------------------------------- #


def test_tool_call_only_reply_routes_to_dispatch() -> None:
    call = ToolCall(id="1", name="open_app", arguments={"name": "spotify"})
    out = Router().route(_r(content=None, tool_calls=[call], finish_reason="tool_calls"))
    assert out.tool_calls == [call]
    assert not out.has_speech
    assert out.has_tool_calls
    # Tool calls present means the turn isn't terminal — the loop must
    # feed results back into the model.
    assert not out.is_terminal


def test_multiple_tool_calls_preserve_order() -> None:
    a = ToolCall(id="1", name="open_app", arguments={"name": "spotify"})
    b = ToolCall(id="2", name="media_control", arguments={"action": "play_pause"})
    out = Router().route(_r(content=None, tool_calls=[a, b]))
    assert out.tool_calls == [a, b]


# --------------------------------------------------------------------------- #
# Mixed replies (speech + tool calls)                                         #
# --------------------------------------------------------------------------- #


def test_text_with_tool_calls_routes_to_both() -> None:
    call = ToolCall(id="1", name="open_app", arguments={"name": "chrome"})
    out = Router().route(
        _r(content="Opening Chrome.", tool_calls=[call], finish_reason="tool_calls")
    )
    assert out.speech == "Opening Chrome."
    assert out.tool_calls == [call]
    assert out.has_speech and out.has_tool_calls
    assert not out.is_terminal


# --------------------------------------------------------------------------- #
# RouteResult helpers                                                         #
# --------------------------------------------------------------------------- #


def test_route_result_is_terminal_when_no_tool_calls() -> None:
    assert RouteResult(speech="done").is_terminal
    assert RouteResult().is_terminal


def test_route_result_not_terminal_when_tool_calls_present() -> None:
    call = ToolCall(id="1", name="x", arguments={})
    assert not RouteResult(tool_calls=[call]).is_terminal


def test_route_result_defaults_empty() -> None:
    r = RouteResult()
    assert r.speech == ""
    assert r.tool_calls == []
    assert not r.has_speech
    assert not r.has_tool_calls
