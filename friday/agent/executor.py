"""Run one ``ToolCall`` through the safety gate and return its result.

The executor is the single chokepoint where every proposed action
either runs for real, runs in dry-run mode, gets bounced to the user
for confirmation, or is denied. The :class:`~friday.agent.safety.SafetyGate`
makes the decision; this module turns the decision into a
:class:`ChatMessage` the LLM can read back as a tool result.

Phase 1 contract
----------------
* Every call returns a ``role="tool"`` ``ChatMessage`` — never raises.
* A successful run audits ``skill_executed``; a runtime exception
  audits ``skill_failed`` and the error string ships back to the LLM.
* A non-``allow`` decision (dry-run, needs_confirmation, deny) writes
  a synthetic tool result so the model knows what happened — the
  audit trail already captured the decision separately.

Phase 3 will extend the ``needs_confirmation`` path with an actual
user-confirmation prompt; for now it surfaces to the LLM as "I can't
do that without a confirmation."
"""

from __future__ import annotations

import logging

from friday.agent.safety import SafetyDecision, SafetyGate
from friday.llm.base import ChatMessage, ToolCall
from friday.skills.registry import SkillRegistry

log = logging.getLogger(__name__)


class Executor:
    """Run a ``ToolCall`` through :class:`SafetyGate` and return its result."""

    def __init__(self, registry: SkillRegistry, gate: SafetyGate) -> None:
        self._registry = registry
        self._gate = gate

    def run(self, call: ToolCall) -> ChatMessage:
        """Evaluate, execute (or not), and package the result for the LLM."""
        decision = self._gate.evaluate(call.name, call.arguments)

        if decision.allow_execution:
            content = self._execute(call)
        else:
            content = _synthetic_result(call, decision)

        return ChatMessage(
            role="tool",
            content=content,
            tool_call_id=call.id,
            name=call.name,
        )

    def _execute(self, call: ToolCall) -> str:
        try:
            result = self._registry.dispatch(call.name, call.arguments)
        except Exception as exc:  # noqa: BLE001 — skill bodies may raise anything
            err = f"{type(exc).__name__}: {exc}"
            self._gate.record_execution(
                call.name, call.arguments, error=err
            )
            log.warning("skill %s raised %s", call.name, err)
            return f"[error] {err}"

        self._gate.record_execution(
            call.name, call.arguments, result=result
        )
        # The LLM gets a string back — most skills already return one,
        # but normalise so downstream serialization doesn't surprise us.
        return str(result) if result is not None else "ok"


# --------------------------------------------------------------------------- #
# Synthetic tool results for non-executed paths                               #
# --------------------------------------------------------------------------- #


def _synthetic_result(call: ToolCall, decision: SafetyDecision) -> str:
    """Render a non-``allow`` decision as text the LLM can react to.

    We word these so the model can carry on and produce a sensible
    verbal response without having to know about safety internals.
    """
    if decision.kind == "dry_run":
        # In dry-run we want the model to behave as if the action
        # succeeded — that's what dry-run is for: rehearsing the full
        # conversation flow without side effects.
        return (
            f"[dry_run] {call.name}({_fmt_args(call.arguments)}) "
            "would have executed; treat it as having succeeded."
        )

    if decision.kind == "needs_confirmation":
        return (
            f"[needs_confirmation] {call.name} was blocked: "
            f"{decision.reason}. Tell the user what you wanted to do "
            "and that you need their confirmation to proceed."
        )

    if decision.kind == "deny":
        return (
            f"[denied] {call.name} was refused: {decision.reason}. "
            "Acknowledge this to the user and pick a different approach."
        )

    # Defensive — every kind handled above.
    return f"[unknown decision: {decision.kind}] {decision.reason}"


def _fmt_args(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in arguments.items())


__all__ = ["Executor"]
