"""Tests for ``friday.agent.executor.Executor``.

The executor's contract: every call produces a ``role="tool"``
ChatMessage tagged with the original tool_call_id, regardless of
whether the skill ran for real, dry-ran, was denied, or raised.
"""

from __future__ import annotations

import pytest

from friday.agent.executor import Executor
from friday.agent.safety import SafetyGate
from friday.config import SafetyConfig
from friday.llm.base import ToolCall
from friday.skills.registry import SkillRegistry, skill


@pytest.fixture
def registry() -> SkillRegistry:
    reg = SkillRegistry()

    @skill(registry=reg)
    def echo(text: str) -> str:
        """Echo back the text."""
        return text.upper()

    @skill(registry=reg)
    def raises(reason: str) -> str:
        """Always raise."""
        raise ValueError(reason)

    @skill(registry=reg, destructive=True)
    def boom(text: str) -> str:
        """Destructive."""
        return text

    @skill(registry=reg, name="open_app")
    def fake_open_app(name: str) -> str:
        """Open an app."""
        return f"launched {name}"

    return reg


def _gate(reg: SkillRegistry, *, dry_run: bool, apps: list[str] | None = None) -> SafetyGate:
    cfg = SafetyConfig(
        dry_run=dry_run,
        app_allowlist=apps if apps is not None else ["spotify"],
        shell_allowlist=[],
    )
    return SafetyGate(cfg, reg)


# --------------------------------------------------------------------------- #
# Allow path                                                                  #
# --------------------------------------------------------------------------- #


def test_allowed_skill_runs_and_returns_result(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=False))
    result = executor.run(ToolCall(id="t1", name="echo", arguments={"text": "hi"}))

    assert result.role == "tool"
    assert result.tool_call_id == "t1"
    assert result.name == "echo"
    assert result.content == "HI"


def test_skill_exception_is_captured_as_error(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=False))
    result = executor.run(
        ToolCall(id="t2", name="raises", arguments={"reason": "nope"})
    )
    assert result.role == "tool"
    assert "[error]" in (result.content or "")
    assert "nope" in (result.content or "")


# --------------------------------------------------------------------------- #
# Dry-run path                                                                #
# --------------------------------------------------------------------------- #


def test_dry_run_returns_synthetic_succeeded_result(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=True))
    result = executor.run(
        ToolCall(id="t3", name="echo", arguments={"text": "hi"})
    )
    content = result.content or ""
    assert "[dry_run]" in content
    assert "echo" in content
    assert "succeeded" in content


def test_destructive_in_dry_run_still_dry_runs(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=True))
    result = executor.run(
        ToolCall(id="t4", name="boom", arguments={"text": "x"})
    )
    assert "[dry_run]" in (result.content or "")


# --------------------------------------------------------------------------- #
# Needs-confirmation path                                                     #
# --------------------------------------------------------------------------- #


def test_needs_confirmation_is_reported_to_llm(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=False))
    result = executor.run(
        ToolCall(id="t5", name="boom", arguments={"text": "x"})
    )
    content = result.content or ""
    assert "[needs_confirmation]" in content
    assert "destructive" in content.lower()


def test_off_allowlist_app_needs_confirmation(registry: SkillRegistry) -> None:
    executor = Executor(
        registry,
        _gate(registry, dry_run=False, apps=["spotify"]),
    )
    result = executor.run(
        ToolCall(id="t6", name="open_app", arguments={"name": "notion"})
    )
    content = result.content or ""
    assert "[needs_confirmation]" in content
    assert "notion" in content


# --------------------------------------------------------------------------- #
# Deny path                                                                   #
# --------------------------------------------------------------------------- #


def test_unknown_skill_is_reported_to_llm(registry: SkillRegistry) -> None:
    executor = Executor(registry, _gate(registry, dry_run=True))
    result = executor.run(
        ToolCall(id="t7", name="nope", arguments={})
    )
    content = result.content or ""
    assert "[denied]" in content
    assert "unknown" in content.lower()
