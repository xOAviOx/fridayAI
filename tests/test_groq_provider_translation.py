"""Translation tests for ``friday.llm.groq_provider``.

The provider's job is two-way translation between FRIDAY's
``ChatMessage`` / ``ToolCall`` shapes and the OpenAI chat-completions
wire format. These tests exercise the translation functions directly,
without instantiating ``GroqLLM`` (which would require the openai
SDK installed and an API key).

Network-touching tests are intentionally out of scope here — they
belong in a separate, opt-in integration suite, not on every push.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from friday.llm.base import ChatMessage, ToolCall
from friday.llm.groq_provider import (
    _estimate_request_tokens,
    _from_openai_response,
    _retry_after_seconds,
    _to_openai_message,
)


# --------------------------------------------------------------------------- #
# ChatMessage → OpenAI dict                                                   #
# --------------------------------------------------------------------------- #


def test_user_message_translates_simply() -> None:
    out = _to_openai_message(ChatMessage(role="user", content="hi"))
    assert out == {"role": "user", "content": "hi"}


def test_system_message_translates_simply() -> None:
    out = _to_openai_message(ChatMessage(role="system", content="be brief"))
    assert out == {"role": "system", "content": "be brief"}


def test_assistant_plain_text_translates_simply() -> None:
    out = _to_openai_message(ChatMessage(role="assistant", content="ok"))
    assert out == {"role": "assistant", "content": "ok"}


def test_assistant_with_tool_calls_uses_openai_shape() -> None:
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="call_1", name="open_app", arguments={"name": "spotify"}),
        ],
    )
    out = _to_openai_message(msg)

    assert out["role"] == "assistant"
    assert out["content"] is None
    assert len(out["tool_calls"]) == 1

    tc = out["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "open_app"
    # Arguments MUST be a JSON string per the OpenAI wire format,
    # not a parsed dict.
    assert tc["function"]["arguments"] == json.dumps({"name": "spotify"})


def test_tool_result_message_includes_tool_call_id() -> None:
    out = _to_openai_message(
        ChatMessage(
            role="tool",
            content="launched spotify",
            tool_call_id="call_1",
            name="open_app",
        )
    )
    assert out == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "launched spotify",
        "name": "open_app",
    }


def test_tool_result_without_tool_call_id_raises() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        _to_openai_message(ChatMessage(role="tool", content="ok"))


# --------------------------------------------------------------------------- #
# OpenAI response → ChatResponse                                              #
# --------------------------------------------------------------------------- #


def _fake_response(
    *,
    content: str | None = "ok",
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    usage: SimpleNamespace | None = None,
    resp_id: str = "resp_123",
    model: str = "llama-3.3-70b-versatile",
) -> SimpleNamespace:
    """Build the minimum OpenAI-SDK-shaped response object we use."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, id=resp_id, model=model)


def test_plain_text_response_round_trips() -> None:
    parsed = _from_openai_response(_fake_response(content="hello there"))
    assert parsed.content == "hello there"
    assert parsed.tool_calls == []
    assert parsed.finish_reason == "stop"


def test_tool_call_arguments_are_json_decoded() -> None:
    fake_tc = SimpleNamespace(
        id="call_42",
        function=SimpleNamespace(
            name="open_app",
            arguments=json.dumps({"name": "chrome"}),
        ),
    )
    parsed = _from_openai_response(
        _fake_response(content=None, tool_calls=[fake_tc]),
    )

    assert parsed.content is None
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.id == "call_42"
    assert tc.name == "open_app"
    # Already-decoded dict — never a JSON string.
    assert tc.arguments == {"name": "chrome"}


def test_tool_call_malformed_json_falls_back_to_raw() -> None:
    fake_tc = SimpleNamespace(
        id="call_bad",
        function=SimpleNamespace(name="x", arguments="{not json"),
    )
    parsed = _from_openai_response(
        _fake_response(content=None, tool_calls=[fake_tc]),
    )
    assert parsed.tool_calls[0].arguments == {"_raw": "{not json"}


def test_usage_translates_prompt_completion_total() -> None:
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=None,
    )
    parsed = _from_openai_response(_fake_response(usage=usage))
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens == 100
    assert parsed.usage.completion_tokens == 20
    assert parsed.usage.total_tokens == 120
    assert parsed.usage.cached_prompt_tokens == 0


def test_usage_picks_up_cached_tokens_when_present() -> None:
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    parsed = _from_openai_response(_fake_response(usage=usage))
    assert parsed.usage is not None
    assert parsed.usage.cached_prompt_tokens == 40


# --------------------------------------------------------------------------- #
# Estimator                                                                   #
# --------------------------------------------------------------------------- #


def test_estimate_request_tokens_sums_content_and_tools() -> None:
    messages = [
        ChatMessage(role="user", content="hello world"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="1", name="open_app", arguments={"name": "spotify"}),
            ],
        ),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open an app.",
                "parameters": {"type": "object"},
            },
        }
    ]
    n = _estimate_request_tokens(messages, tools)
    # Floor is len // 4 ≥ 1; the call definitely produces more than zero.
    assert n > 0


# --------------------------------------------------------------------------- #
# _retry_after_seconds                                                        #
# --------------------------------------------------------------------------- #


class _FakeExc(Exception):
    def __init__(self, status: int | None = 429, retry_after: str | None = "2.5") -> None:
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = SimpleNamespace(status_code=status, headers=headers)


def test_retry_after_seconds_picks_up_value() -> None:
    assert _retry_after_seconds(_FakeExc(retry_after="3.0")) == 3.0


def test_retry_after_seconds_returns_none_for_non_429() -> None:
    assert _retry_after_seconds(_FakeExc(status=500)) is None


def test_retry_after_seconds_returns_none_when_header_missing() -> None:
    assert _retry_after_seconds(_FakeExc(retry_after=None)) is None


def test_retry_after_seconds_returns_none_for_non_http_exception() -> None:
    assert _retry_after_seconds(ValueError("not http")) is None
