"""Tests for the safety gate.

Covers the four policy axes (unknown skill, app allowlist, destructive
flag, dry-run) plus the shell-allowlist surface and the audit-log
contract. Tests build a fresh :class:`SkillRegistry` per test so they
never depend on the process-wide ``default_registry``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from friday.agent.safety import SafetyDecision, SafetyGate
from friday.config import SafetyConfig
from friday.skills.registry import SkillRegistry, skill
from friday.utils.logging import setup_logging


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def registry() -> SkillRegistry:
    """A fresh registry with one safe and one destructive skill plus open_app."""
    reg = SkillRegistry()

    @skill(registry=reg)
    def safe(text: str) -> str:
        """A safe skill."""
        return text

    @skill(registry=reg, destructive=True)
    def boom(text: str) -> str:
        """A destructive skill."""
        return text

    # Real builtins call into the OS — we re-declare an open_app stand-in
    # here so the gate's allowlist check has something to bind against
    # under our own controlled registry.
    @skill(registry=reg, name="open_app")
    def fake_open_app(name: str) -> str:
        """Open the named app."""
        return name

    return reg


def _config(
    *,
    dry_run: bool = True,
    apps: list[str] | None = None,
    shells: list[str] | None = None,
) -> SafetyConfig:
    return SafetyConfig(
        dry_run=dry_run,
        app_allowlist=apps if apps is not None else ["spotify", "chrome"],
        shell_allowlist=shells if shells is not None else ["echo", "ls"],
    )


# --------------------------------------------------------------------------- #
# Skill evaluation                                                            #
# --------------------------------------------------------------------------- #


def test_unknown_skill_is_denied(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(), registry)
    decision = gate.evaluate("nope", {})
    assert decision.kind == "deny"
    assert "unknown skill" in decision.reason
    assert not decision.allow_execution


def test_safe_skill_in_dry_run_yields_dry_run(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(dry_run=True), registry)
    decision = gate.evaluate("safe", {"text": "hi"})
    assert decision.kind == "dry_run"
    assert not decision.allow_execution


def test_safe_skill_outside_dry_run_yields_allow(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(dry_run=False), registry)
    decision = gate.evaluate("safe", {"text": "hi"})
    assert decision.kind == "allow"
    assert decision.allow_execution


def test_destructive_skill_outside_dry_run_needs_confirmation(
    registry: SkillRegistry,
) -> None:
    gate = SafetyGate(_config(dry_run=False), registry)
    decision = gate.evaluate("boom", {"text": "hi"})
    assert decision.kind == "needs_confirmation"
    assert "destructive" in decision.reason
    assert not decision.allow_execution


def test_destructive_skill_in_dry_run_still_flows_through(
    registry: SkillRegistry,
) -> None:
    # Dry-run is exactly when destructive should be allowed to *log*
    # an intent — there are no side effects to confirm.
    gate = SafetyGate(_config(dry_run=True), registry)
    decision = gate.evaluate("boom", {"text": "hi"})
    assert decision.kind == "dry_run"


# ----- App allowlist --------------------------------------------------------


def test_allowlisted_app_passes_to_dry_run(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(apps=["spotify"], dry_run=True), registry)
    decision = gate.evaluate("open_app", {"name": "spotify"})
    assert decision.kind == "dry_run"


def test_allowlisted_app_runs_outside_dry_run(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(apps=["spotify"], dry_run=False), registry)
    decision = gate.evaluate("open_app", {"name": "spotify"})
    assert decision.kind == "allow"


def test_off_allowlist_app_needs_confirmation(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(apps=["spotify"]), registry)
    decision = gate.evaluate("open_app", {"name": "notion"})
    assert decision.kind == "needs_confirmation"
    assert "notion" in decision.reason
    assert decision.detail["app"] == "notion"
    assert decision.detail["allowlist"] == ["spotify"]


def test_app_allowlist_matching_is_case_insensitive(
    registry: SkillRegistry,
) -> None:
    gate = SafetyGate(_config(apps=["Spotify"]), registry)
    decision = gate.evaluate("open_app", {"name": "SPOTIFY"})
    assert decision.kind == "dry_run"


def test_open_app_without_name_is_denied(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(), registry)
    decision = gate.evaluate("open_app", {"name": ""})
    assert decision.kind == "deny"
    assert "without a name" in decision.reason


# --------------------------------------------------------------------------- #
# Shell evaluation                                                            #
# --------------------------------------------------------------------------- #


def test_allowlisted_shell_dry_runs(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(shells=["echo"], dry_run=True), registry)
    assert gate.evaluate_shell("echo hello").kind == "dry_run"


def test_allowlisted_shell_runs_outside_dry_run(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(shells=["echo"], dry_run=False), registry)
    assert gate.evaluate_shell("echo hello").kind == "allow"


def test_off_allowlist_shell_needs_confirmation(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(shells=["echo"]), registry)
    decision = gate.evaluate_shell("rm -rf /")
    assert decision.kind == "needs_confirmation"
    assert "rm" in decision.reason


def test_empty_shell_command_is_denied(registry: SkillRegistry) -> None:
    gate = SafetyGate(_config(), registry)
    assert gate.evaluate_shell("   ").kind == "deny"


# --------------------------------------------------------------------------- #
# Audit log                                                                   #
# --------------------------------------------------------------------------- #


def test_evaluate_writes_jsonl_audit_entry(
    registry: SkillRegistry, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.log"
    setup_logging(level="WARNING", audit_path=audit_path)

    gate = SafetyGate(_config(), registry)
    gate.evaluate("safe", {"text": "hi"})

    line = audit_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "skill_invoke"
    assert payload["skill"] == "safe"
    assert payload["arguments"] == {"text": "hi"}
    assert payload["decision"] == "dry_run"


def test_record_execution_writes_separate_audit_event(
    registry: SkillRegistry, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.log"
    setup_logging(level="WARNING", audit_path=audit_path)

    gate = SafetyGate(_config(dry_run=False), registry)
    gate.record_execution("safe", {"text": "hi"}, result="hi")

    line = audit_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "skill_executed"
    assert payload["result"] == "hi"
    assert payload["error"] is None


def test_record_execution_with_error_logs_as_failed(
    registry: SkillRegistry, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.log"
    setup_logging(level="WARNING", audit_path=audit_path)

    gate = SafetyGate(_config(dry_run=False), registry)
    gate.record_execution("safe", {"text": "hi"}, error="boom")

    line = audit_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "skill_failed"
    assert payload["error"] == "boom"


# --------------------------------------------------------------------------- #
# Decision constructors                                                       #
# --------------------------------------------------------------------------- #


def test_decision_constructors_set_kind_and_detail() -> None:
    d = SafetyDecision.allow()
    assert d.kind == "allow" and d.allow_execution

    d = SafetyDecision.dry_run("custom reason")
    assert d.kind == "dry_run" and d.reason == "custom reason"

    d = SafetyDecision.needs_confirmation("nope", app="x")
    assert d.kind == "needs_confirmation"
    assert d.detail == {"app": "x"}

    d = SafetyDecision.deny("no")
    assert d.kind == "deny" and not d.allow_execution
