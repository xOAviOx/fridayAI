"""Tests for ``friday.llm.history.SlidingWindowHistory``.

The window is anchored on user messages; tool-call/tool-result pairs
tag along with their parent turn and should never be split.
"""

from __future__ import annotations

import pytest

from friday.llm.base import ChatMessage, ToolCall
from friday.llm.history import SlidingWindowHistory


def _msg(role: str, content: str | None = None, **kwargs: object) -> ChatMessage:
    return ChatMessage(role=role, content=content, **kwargs)  # type: ignore[arg-type]


def test_system_message_is_always_present() -> None:
    h = SlidingWindowHistory(system="you are friday")
    h.add(_msg("user", "hi"))

    out = h.messages()
    assert out[0].role == "system"
    assert out[0].content == "you are friday"


def test_no_system_message_means_only_history() -> None:
    h = SlidingWindowHistory()
    h.add(_msg("user", "hi"))
    h.add(_msg("assistant", "hello"))
    out = h.messages()
    assert [m.role for m in out] == ["user", "assistant"]


def test_window_keeps_last_n_user_turns() -> None:
    h = SlidingWindowHistory(system="s", max_user_turns=2)
    for i in range(5):
        h.add(_msg("user", f"u{i}"))
        h.add(_msg("assistant", f"a{i}"))

    out = h.messages()
    contents = [m.content for m in out]
    # System + last two (user, assistant) pairs.
    assert contents == ["s", "u3", "a3", "u4", "a4"]


def test_tool_call_and_result_stay_paired_with_their_user_turn() -> None:
    h = SlidingWindowHistory(system="s", max_user_turns=1)
    # Old turn — should be evicted by the new one.
    h.add(_msg("user", "u0"))
    h.add(_msg("assistant", "a0"))
    # New turn with a tool-call dance.
    h.add(_msg("user", "u1"))
    h.add(
        _msg(
            "assistant",
            None,
            tool_calls=[ToolCall(id="1", name="open_app", arguments={"name": "spotify"})],
        )
    )
    h.add(_msg("tool", "launched spotify", tool_call_id="1", name="open_app"))
    h.add(_msg("assistant", "done"))

    out = h.messages()
    roles = [m.role for m in out]
    # System + the entire new turn (user, assistant w/ tool_calls,
    # tool result, assistant final). The old turn is dropped.
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert out[1].content == "u1"


def test_extend_appends_in_order() -> None:
    h = SlidingWindowHistory()
    h.extend([_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")])
    assert [m.content for m in h.messages()] == ["a", "b", "c"]


def test_adding_system_message_through_add_is_rejected() -> None:
    h = SlidingWindowHistory()
    with pytest.raises(ValueError, match="system message can't be added"):
        h.add(_msg("system", "no"))


def test_system_message_must_have_system_role() -> None:
    with pytest.raises(ValueError, match="role='system'"):
        SlidingWindowHistory(system=_msg("user", "wrong role"))


def test_reset_drops_history_but_keeps_system() -> None:
    h = SlidingWindowHistory(system="s")
    h.add(_msg("user", "hi"))
    h.add(_msg("assistant", "bye"))
    h.reset()
    out = h.messages()
    assert [m.role for m in out] == ["system"]


def test_max_user_turns_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_user_turns"):
        SlidingWindowHistory(max_user_turns=0)


def test_len_returns_windowed_size() -> None:
    h = SlidingWindowHistory(system="s", max_user_turns=2)
    for i in range(5):
        h.add(_msg("user", f"u{i}"))
        h.add(_msg("assistant", f"a{i}"))
    # System + (u3, a3, u4, a4) = 5 messages.
    assert len(h) == 5
