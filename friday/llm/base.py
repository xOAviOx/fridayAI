"""LLM provider interface.

The data classes here are intentionally close to the OpenAI chat-completion
shape because Groq is OpenAI-SDK compatible and most other providers
(Gemini, Anthropic) can be adapted to/from it.

Phase 0: definition only. Implementations land in Phase 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants the agent to execute.

    ``arguments`` is the already-parsed dict — providers are responsible
    for handling the JSON-string-vs-dict quirk so the agent never sees it.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    """One message in a chat thread.

    For ``role="tool"`` messages, ``tool_call_id`` and ``name`` must be
    set so the model can match the result back to its original tool call.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Cached input tokens (don't count toward Groq TPM). Providers that
    # don't report cache stats leave this at 0.
    cached_prompt_tokens: int = 0


@dataclass
class ChatResponse:
    """Result of a single non-streaming LLM call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    # Provider-specific metadata (model name, request id, etc.) that's
    # useful for debugging but the agent shouldn't depend on.
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Chat completion with tool-calling support.

    Streaming is added in Phase 2 by introducing a separate ``stream_chat``
    method that yields ``ChatResponse`` deltas, not by changing this one.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send ``messages`` and (optional) ``tools`` to the model."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The currently selected model id (for logging / token accounting)."""

    def close(self) -> None:  # pragma: no cover - default is no-op
        """Release any underlying resources. Default is a no-op."""
