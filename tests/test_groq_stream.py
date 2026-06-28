"""Unit tests for GroqLLM streaming — translation layer only, no real API.

We feed synthetic chunk objects that mirror the OpenAI streaming wire
format and verify that StreamedResponse assembles them correctly into
a ChatResponse with the right text, tool calls, and usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from friday.llm.groq_provider import StreamedResponse, _assemble_response, _parse_usage
from friday.llm.base import ChatResponse, ToolCall, TokenUsage


# --------------------------------------------------------------------------- #
# Fake OpenAI streaming objects                                               #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeFn:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _FakeToolCallDelta:
    index: int
    id: str | None = None
    function: _FakeFn | None = None


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[_FakeToolCallDelta] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None


@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15
    prompt_tokens_details: Any = None


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


def _text_chunk(text: str, finish: str | None = None) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeChoice(_FakeDelta(content=text), finish_reason=finish)])


def _tool_chunk(
    index: int,
    id_: str | None = None,
    name: str | None = None,
    args: str | None = None,
    finish: str | None = None,
) -> _FakeChunk:
    fn = _FakeFn(name=name, arguments=args)
    tc = _FakeToolCallDelta(index=index, id=id_, function=fn)
    return _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[tc]), finish_reason=finish)])


def _usage_chunk(prompt: int = 10, completion: int = 5) -> _FakeChunk:
    """Final usage-only chunk (empty choices list)."""
    return _FakeChunk(choices=[], usage=_FakeUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    ))


# --------------------------------------------------------------------------- #
# Tests — text-only stream                                                    #
# --------------------------------------------------------------------------- #


class TestTextStream:
    def _make(self, chunks):
        return StreamedResponse(iter(chunks))

    def test_yields_all_text_tokens(self):
        chunks = [
            _text_chunk("Hello"),
            _text_chunk(", "),
            _text_chunk("world!"),
            _text_chunk(None, finish="stop"),
        ]
        sr = self._make(chunks)
        tokens = list(sr)
        assert tokens == ["Hello", ", ", "world!"]

    def test_response_has_assembled_content(self):
        chunks = [
            _text_chunk("Hi "),
            _text_chunk("there."),
            _text_chunk(None, finish="stop"),
        ]
        sr = self._make(chunks)
        list(sr)  # exhaust
        assert sr.response.content == "Hi there."
        assert sr.response.tool_calls == []
        assert sr.response.finish_reason == "stop"

    def test_response_accessible_without_explicit_drain(self):
        chunks = [_text_chunk("Yo.", finish="stop")]
        sr = self._make(chunks)
        # Access .response without iterating first — should auto-drain.
        assert sr.response.content == "Yo."

    def test_empty_content_gives_none(self):
        chunks = [_FakeChunk(choices=[_FakeChoice(_FakeDelta(content=None), finish_reason="stop")])]
        sr = self._make(chunks)
        list(sr)
        assert sr.response.content is None

    def test_usage_captured_from_final_chunk(self):
        chunks = [
            _text_chunk("Done.", finish="stop"),
            _usage_chunk(prompt=20, completion=8),
        ]
        sr = self._make(chunks)
        list(sr)
        usage = sr.response.usage
        assert usage is not None
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 28

    def test_iterate_twice_is_noop(self):
        chunks = [_text_chunk("Hey.", finish="stop")]
        sr = self._make(chunks)
        first = list(sr)
        second = list(sr)  # should not re-iterate the already-exhausted stream
        assert first == ["Hey."]
        assert second == []


# --------------------------------------------------------------------------- #
# Tests — tool-call stream                                                    #
# --------------------------------------------------------------------------- #


class TestToolCallStream:
    def _make(self, chunks):
        return StreamedResponse(iter(chunks))

    def test_single_tool_call_assembled(self):
        chunks = [
            _tool_chunk(0, id_="call_abc", name="open_app"),
            _tool_chunk(0, args='{"name":'),
            _tool_chunk(0, args='"spotify"}', finish="tool_calls"),
        ]
        sr = self._make(chunks)
        list(sr)
        assert len(sr.response.tool_calls) == 1
        tc = sr.response.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "open_app"
        assert tc.arguments == {"name": "spotify"}

    def test_two_parallel_tool_calls(self):
        chunks = [
            _tool_chunk(0, id_="c1", name="open_app"),
            _tool_chunk(1, id_="c2", name="web_search"),
            _tool_chunk(0, args='{"name":"spotify"}'),
            _tool_chunk(1, args='{"query":"weather"}', finish="tool_calls"),
        ]
        sr = self._make(chunks)
        list(sr)
        assert len(sr.response.tool_calls) == 2
        by_name = {tc.name: tc for tc in sr.response.tool_calls}
        assert by_name["open_app"].arguments == {"name": "spotify"}
        assert by_name["web_search"].arguments == {"query": "weather"}

    def test_finish_reason_is_tool_calls(self):
        chunks = [
            _tool_chunk(0, id_="x", name="system_info", args='{"metric":"time"}',
                        finish="tool_calls"),
        ]
        sr = self._make(chunks)
        list(sr)
        assert sr.response.finish_reason == "tool_calls"

    def test_text_plus_tool_call(self):
        """Model can emit text AND tool calls in the same response."""
        chunks = [
            _text_chunk("Sure, let me check."),
            _tool_chunk(0, id_="t1", name="system_info", args='{"metric":"time"}',
                        finish="tool_calls"),
        ]
        sr = self._make(chunks)
        tokens = list(sr)
        assert tokens == ["Sure, let me check."]
        assert sr.response.content == "Sure, let me check."
        assert len(sr.response.tool_calls) == 1

    def test_bad_json_args_fall_back_gracefully(self):
        chunks = [
            _tool_chunk(0, id_="bad", name="open_app", args="{INVALID", finish="tool_calls"),
        ]
        sr = self._make(chunks)
        list(sr)
        tc = sr.response.tool_calls[0]
        # Should not raise — falls back to {"_raw": ...}
        assert "_raw" in tc.arguments


# --------------------------------------------------------------------------- #
# Tests — _assemble_response and _parse_usage helpers                         #
# --------------------------------------------------------------------------- #


class TestAssembleResponse:
    def test_empty_produces_minimal_response(self):
        r = _assemble_response("", [], "stop", None)
        assert r.content is None
        assert r.tool_calls == []
        assert r.finish_reason == "stop"

    def test_text_becomes_content(self):
        r = _assemble_response("hello", [], "stop", None)
        assert r.content == "hello"

    def test_tool_calls_parsed(self):
        raw = [{"id": "c1", "name": "open_app", "args": '{"name":"code"}'}]
        r = _assemble_response("", raw, "tool_calls", None)
        assert r.tool_calls[0].name == "open_app"
        assert r.tool_calls[0].arguments == {"name": "code"}


class TestParseUsage:
    def test_basic_fields(self):
        raw = _FakeUsage(prompt_tokens=15, completion_tokens=7, total_tokens=22)
        u = _parse_usage(raw)
        assert u.prompt_tokens == 15
        assert u.completion_tokens == 7
        assert u.total_tokens == 22
        assert u.cached_prompt_tokens == 0
