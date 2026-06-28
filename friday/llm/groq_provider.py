"""Groq chat-completion LLM provider with tool calling and streaming.

Groq is OpenAI-SDK compatible — the wire format, the tool-call shape,
the usage payload all match — so we drive it through the ``openai``
Python client with a custom ``base_url``. Same trick as
:mod:`friday.stt.groq_whisper`; same dependency.

What this module owns
---------------------
* Translation in both directions: :class:`~friday.llm.base.ChatMessage`
  / :class:`~friday.llm.base.ToolCall` ↔ the OpenAI chat-completions
  payload. The agent loop never sees a raw OpenAI dict and never has
  to JSON-decode tool-call arguments by hand.
* Free-tier discipline: a :class:`~friday.utils.tokens.BudgetTracker`
  guards every request, blocks on RPM/TPM exhaustion, and honors
  ``retry-after`` on 429s through bounded retries.
* Stateless ``chat()`` — the sliding window lives in
  :class:`~friday.llm.history.SlidingWindowHistory`, owned by the
  agent loop. The provider just sends whatever it gets.
* Phase 2: ``stream_chat()`` returns a :class:`StreamedResponse` whose
  ``__iter__`` yields ``str`` text deltas.  Calling ``.response`` on it
  (or exhausting the iterator) produces the full :class:`ChatResponse`
  including any tool calls that came in over the stream.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from friday.llm.base import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    ToolCall,
    TokenUsage,
)
from friday.utils.tokens import BudgetTracker, estimate_tokens

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0
_MAX_RETRIES = 3

# Regex that extracts <tool_call>…</tool_call> blocks from model text.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_PROMPT_TOOLS_INSTRUCTION = """
You have access to tools. Call a tool by outputting EXACTLY this format (nothing else on that line):
<tool_call>{{"name": "TOOL_NAME", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}</tool_call>

You may call one tool per response. After each tool result you'll be called again. Once all actions are done, reply with a short spoken confirmation.

Available tools (JSON schemas):
{schemas}
""".strip()


class GroqLLM(LLMProvider):
    """Groq chat completion via the OpenAI-SDK-compatible endpoint.

    Parameters
    ----------
    api_key:
        Groq API key (loaded by :mod:`friday.config` from
        ``GROQ_API_KEY``).
    model:
        Groq model id. ``llama-3.3-70b-versatile`` is the brief's
        default; switch via ``config.yaml``.
    base_url:
        OpenAI-SDK base URL. Default is Groq's; overridable for tests.
    timeout_s:
        Per-request HTTP timeout.
    budget:
        Optional :class:`BudgetTracker`. When provided, every call
        passes through the tracker for RPM/TPM gating and 429 handling.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        budget: BudgetTracker | None = None,
        prompt_tools: bool = False,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GroqLLM requires a non-empty api_key")

        # Lazy import so Phase 0 boot doesn't need the LLM extras.
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Groq LLM selected but the `openai` package isn't installed. "
                "Install the LLM extras:\n"
                "    uv sync --extra llm-groq\n"
                "or:\n"
                "    pip install -e '.[llm-groq]'"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )
        self._model = model
        self._budget = budget
        self._prompt_tools = prompt_tools
        log.debug(
            "GroqLLM ready: model=%s base_url=%s budget=%s",
            model,
            base_url,
            budget is not None,
        )

    @property
    def model(self) -> str:
        return self._model

    # ----- public API ------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat-completion request and return the parsed response."""
        payload_messages = [_to_openai_message(m) for m in messages]
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if tools:
            request_kwargs["tools"] = tools
            # "auto" lets the model decide whether to call a tool; Groq
            # defaults this anyway but we set it explicitly so behavior
            # doesn't drift if the default changes.
            request_kwargs["tool_choice"] = "auto"

        # Cheap pre-flight token estimate so the budget tracker can
        # decide whether we need to sleep before posting.
        estimated = _estimate_request_tokens(messages, tools)
        if self._budget is not None:
            self._budget.before_request(estimated_tokens=estimated)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:  # noqa: BLE001 — SDK class hierarchy varies
                last_exc = exc
                retry_after = _retry_after_seconds(exc)
                if retry_after is None or attempt == _MAX_RETRIES:
                    raise
                if self._budget is not None:
                    self._budget.handle_429(retry_after)
                else:
                    # No budget tracker → sleep ourselves; small import
                    # to avoid pulling time at module load.
                    import time

                    time.sleep(max(0.1, retry_after))
                continue

            parsed = _from_openai_response(response)
            if self._budget is not None and parsed.usage is not None:
                self._budget.record(tokens=parsed.usage.total_tokens)
            return parsed

        # Unreachable — the loop above either returns or raises — but
        # mypy can't see that.
        assert last_exc is not None
        raise last_exc

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> "StreamedResponse":
        """Stream a chat response token by token.

        Returns a :class:`StreamedResponse`. Iterate it to get ``str``
        text deltas as they arrive; once exhausted, ``.response`` holds
        the fully assembled :class:`ChatResponse` (including tool calls
        that arrived via the stream).

        Budget accounting fires the ``before_request`` estimate before
        the HTTP call and records actual usage after the stream closes.
        """
        # prompt_tools mode: inject tool schemas as text in the last user
        # message instead of using the native tools API parameter.
        # Used for openai_compat endpoints that silently drop the tools param.
        if self._prompt_tools and tools:
            messages = _inject_prompt_tools(messages, tools)
            tools = None  # don't send native tools

        payload_messages = [_to_openai_message(m) for m in messages]
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": temperature,
            "stream": True,
        }
        # stream_options is Groq / OpenAI specific — only add it for
        # known-compatible endpoints so other providers don't choke on it.
        if "groq.com" in self._client.base_url.host:
            request_kwargs["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        estimated = _estimate_request_tokens(messages, tools)
        if self._budget is not None:
            self._budget.before_request(estimated_tokens=estimated)

        raw_stream = self._client.chat.completions.create(**request_kwargs)
        return StreamedResponse(
            raw_stream, budget=self._budget, prompt_tools=self._prompt_tools
        )

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort
            pass
        self._client = None


# --------------------------------------------------------------------------- #
# Translation: ChatMessage → OpenAI dict                                      #
# --------------------------------------------------------------------------- #


def _to_openai_message(m: ChatMessage) -> dict[str, Any]:
    """Translate one ``ChatMessage`` to an OpenAI chat-completions dict."""
    if m.role == "tool":
        # Tool-result messages need both ``tool_call_id`` (to match the
        # assistant's call) and ``content``. ``name`` is conventional
        # but not required by the wire format.
        if m.tool_call_id is None:
            raise ValueError("role='tool' message requires tool_call_id")
        out: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content or "",
        }
        if m.name:
            out["name"] = m.name
        return out

    if m.role == "assistant" and m.tool_calls:
        # Assistant message that *requests* tool calls. Content is
        # often empty in this shape; we still send it (as null) so the
        # SDK doesn't drop the field.
        return {
            "role": "assistant",
            "content": m.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # OpenAI expects arguments as a JSON *string*,
                        # not a parsed object.
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ],
        }

    return {"role": m.role, "content": m.content or ""}


# --------------------------------------------------------------------------- #
# Translation: OpenAI response → ChatResponse                                 #
# --------------------------------------------------------------------------- #


def _from_openai_response(response: Any) -> ChatResponse:
    """Translate the OpenAI ``ChatCompletion`` object back to ``ChatResponse``."""
    choice = response.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] = []
    raw_calls = getattr(message, "tool_calls", None) or []
    for raw in raw_calls:
        # The SDK exposes a structured object, but the function arguments
        # are still a JSON-encoded string per the OpenAI wire format.
        # Decode here so the agent never sees a string-shaped dict.
        fn = raw.function
        try:
            arguments = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError as exc:
            log.warning(
                "tool call %s arguments aren't valid JSON: %s", raw.id, exc
            )
            arguments = {"_raw": fn.arguments}
        if not isinstance(arguments, dict):
            log.warning(
                "tool call %s arguments decoded to %s, expected dict",
                raw.id,
                type(arguments).__name__,
            )
            arguments = {"_raw": arguments}
        tool_calls.append(ToolCall(id=raw.id, name=fn.name, arguments=arguments))

    usage = None
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        cached = 0
        details = getattr(raw_usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        usage = TokenUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
            cached_prompt_tokens=cached,
        )

    return ChatResponse(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        usage=usage,
        raw={"id": getattr(response, "id", None), "model": getattr(response, "model", None)},
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _inject_prompt_tools(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]],
) -> list[ChatMessage]:
    """Append a compact tool-schema block to the last user message."""
    # Compact schema: keep name, description, and parameters only.
    compact = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters", {}),
        }
        for t in tools
    ]
    schemas_json = json.dumps(compact, indent=2)
    instruction = _PROMPT_TOOLS_INSTRUCTION.format(schemas=schemas_json)

    # Find the last user message and append the tool block to it.
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].role == "user":
            original = result[i].content or ""
            result[i] = ChatMessage(
                role="user",
                content=f"{original}\n\n{instruction}",
            )
            return result

    # No user message found — prepend as a standalone user message.
    result.insert(0, ChatMessage(role="user", content=instruction))
    return result


def _parse_prompt_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract <tool_call> blocks from model text.

    Returns ``(clean_text, tool_calls)`` where ``clean_text`` has the
    blocks removed so only the spoken reply goes to TTS.
    """
    tool_calls: list[ToolCall] = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"_raw": arguments}
            tool_calls.append(ToolCall(id=f"ptc_{i}", name=name, arguments=arguments))
        except json.JSONDecodeError:
            log.warning("prompt tool call has invalid JSON: %r", raw[:120])

    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, tool_calls


def _estimate_request_tokens(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None,
) -> int:
    """Cheap upper bound on the prompt tokens this call will send."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += estimate_tokens(tc.name)
            total += estimate_tokens(json.dumps(tc.arguments))
    if tools:
        # The tool schema is on the wire too; counting it stops the
        # tracker from undershooting when there are lots of tools.
        total += estimate_tokens(json.dumps(tools))
    return total


def _retry_after_seconds(exc: Exception) -> float | None:
    """Mirror of :func:`friday.stt.groq_whisper._retry_after_seconds`.

    Kept duplicated rather than moved to a shared util because the two
    sites have different "give up" semantics — the STT one short-loops
    inside the provider, this one delegates to BudgetTracker.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    if status != 429:
        return None
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Phase 2: Streaming response wrapper                                         #
# --------------------------------------------------------------------------- #


class StreamedResponse:
    """Iterable wrapper around a streaming OpenAI chat-completion.

    Iterate this object to receive ``str`` text deltas as the model
    generates them. Once the iterator is exhausted (or ``.response``
    is accessed), the assembled :class:`ChatResponse` is available —
    including any tool calls that came through the stream.

    Usage::

        streamed = llm.stream_chat(messages, tools=tools)
        for token in streamed:          # text deltas arrive here
            sentence_buffer.push(token)
        response = streamed.response    # fully assembled after iter

    Thread safety: the iterator is not reentrant.  Call it from a
    single thread (the turn worker).  ``.response`` is safe to read
    from any thread *after* the iterator has been exhausted.
    """

    def __init__(
        self, raw_stream: Any, *, budget: Any = None, prompt_tools: bool = False
    ) -> None:
        self._stream = raw_stream
        self._budget = budget
        self._prompt_tools = prompt_tools
        self._response: ChatResponse | None = None
        self._exhausted = False

    def __iter__(self) -> Iterator[str]:
        """Yield text deltas; populate ``.response`` when done."""
        if self._exhausted:
            return

        full_text = ""
        # Accumulate tool-call deltas indexed by their position in the
        # model's tool_calls list.  Each entry is {"id", "name", "args"}.
        raw_tcs: list[dict[str, str]] = []
        finish_reason = "stop"
        usage: TokenUsage | None = None

        for chunk in self._stream:
            # The final usage-only chunk (stream_options.include_usage)
            # arrives with an empty choices list.
            if not chunk.choices:
                raw_u = getattr(chunk, "usage", None)
                if raw_u is not None:
                    usage = _parse_usage(raw_u)
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Text delta — yield immediately so TTS can start.
            text_piece = getattr(delta, "content", None)
            if text_piece:
                full_text += text_piece
                yield text_piece

            # Tool-call deltas — accumulate argument strings.
            tc_deltas = getattr(delta, "tool_calls", None) or []
            for tc_d in tc_deltas:
                idx: int = tc_d.index
                while len(raw_tcs) <= idx:
                    raw_tcs.append({"id": "", "name": "", "args": ""})
                if tc_d.id:
                    raw_tcs[idx]["id"] = tc_d.id
                fn = getattr(tc_d, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        raw_tcs[idx]["name"] = fn.name
                    arg_piece = getattr(fn, "arguments", None)
                    if arg_piece:
                        raw_tcs[idx]["args"] += arg_piece

            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr

        self._exhausted = True

        # prompt_tools mode: extract <tool_call> blocks from the text.
        if self._prompt_tools:
            clean_text, ptc_calls = _parse_prompt_tool_calls(full_text)
            if ptc_calls:
                self._response = ChatResponse(
                    content=clean_text or None,
                    tool_calls=ptc_calls,
                    finish_reason="tool_calls",
                    usage=usage,
                )
            else:
                self._response = _assemble_response(
                    clean_text, raw_tcs, finish_reason, usage
                )
        else:
            self._response = _assemble_response(full_text, raw_tcs, finish_reason, usage)

        if self._budget is not None and usage is not None:
            self._budget.record(tokens=usage.total_tokens)

    @property
    def response(self) -> ChatResponse:
        """The assembled response — drain the iterator first if needed."""
        if not self._exhausted:
            for _ in self:
                pass  # drain
        assert self._response is not None
        return self._response


def _parse_usage(raw_u: Any) -> TokenUsage:
    """Build a TokenUsage from a streaming usage object."""
    cached = 0
    details = getattr(raw_u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return TokenUsage(
        prompt_tokens=getattr(raw_u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw_u, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw_u, "total_tokens", 0) or 0,
        cached_prompt_tokens=cached,
    )


def _assemble_response(
    full_text: str,
    raw_tcs: list[dict[str, str]],
    finish_reason: str,
    usage: TokenUsage | None,
) -> ChatResponse:
    """Build a ChatResponse from the accumulated streaming data."""
    tool_calls: list[ToolCall] = []
    for raw in raw_tcs:
        args_str = raw.get("args", "")
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            log.warning("stream tool call arguments aren't valid JSON: %r", args_str)
            arguments = {"_raw": args_str}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        tool_calls.append(
            ToolCall(id=raw.get("id", ""), name=raw.get("name", ""), arguments=arguments)
        )
    return ChatResponse(
        content=full_text or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


__all__ = ["GroqLLM", "StreamedResponse"]
