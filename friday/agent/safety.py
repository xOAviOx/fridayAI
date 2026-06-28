"""Safety gate: the policy layer in front of every executed action.

Built *before* the agent loop on purpose — the spec is emphatic that
safety is not bolted on at the end. Every Phase 1 tool call routes
through :class:`SafetyGate` first; the gate returns a
:class:`SafetyDecision` and the executor (chunk 6) honors it.

Four policy axes, in evaluation order
-------------------------------------
1. **Unknown skill** → :meth:`SafetyDecision.deny`. The LLM hallucinated
   a tool we don't expose. Never run anything.
2. **App allowlist** (for ``open_app``) → off-allowlist app names
   require explicit confirmation. Apps on the list pass through to the
   later axes.
3. **Destructive flag** (the marker the ``@skill`` decorator stores) →
   if the skill is destructive *and* we're not in dry-run, require
   confirmation. In dry-run, destructive flows through as a dry-run
   entry so the audit log still records the intent.
4. **Dry-run gate** → if ``safety.dry_run`` is on, no side effects
   actually run, regardless of allowlists. The executor turns this
   into a stubbed tool response for the LLM, audited as such.

A separate :meth:`SafetyGate.evaluate_shell` covers any future skill
that shells out (none of the five builtins do, but the surface needs
to exist before chunk 6 ships a code-exec sandbox).

Every decision the gate emits is audit-logged via
:func:`friday.utils.logging.audit` — the audit file is the canonical
record of "what FRIDAY would have done / did" and a Phase 3 review
loop reads it back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from friday.config import SafetyConfig
from friday.skills.registry import SkillRegistry
from friday.utils.logging import audit

log = logging.getLogger(__name__)

# Skill names whose first argument is matched against ``app_allowlist``.
# Hard-coded for Phase 1 — Phase 3 will move this to a registry-level
# annotation so new "open something" skills declare their allowlist
# binding explicitly.
_APP_GATED_SKILLS: frozenset[str] = frozenset({"open_app"})

DecisionKind = Literal["allow", "dry_run", "needs_confirmation", "deny"]


# --------------------------------------------------------------------------- #
# Decision                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SafetyDecision:
    """One policy decision about a proposed tool invocation.

    The executor reads :attr:`allow_execution` to decide whether to
    actually call through. Every kind other than ``allow`` is a
    "did not execute" outcome; the reason explains why.
    """

    kind: DecisionKind
    reason: str = ""
    # Optional structured context — handy for the executor to include
    # in the tool response it returns to the LLM ("I can't open Notion;
    # it isn't on the app allowlist").
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def allow_execution(self) -> bool:
        return self.kind == "allow"

    @classmethod
    def allow(cls, reason: str = "") -> SafetyDecision:
        return cls("allow", reason)

    @classmethod
    def dry_run(cls, reason: str = "dry_run is on") -> SafetyDecision:
        return cls("dry_run", reason)

    @classmethod
    def needs_confirmation(
        cls, reason: str, **detail: Any
    ) -> SafetyDecision:
        return cls("needs_confirmation", reason, detail=dict(detail))

    @classmethod
    def deny(cls, reason: str, **detail: Any) -> SafetyDecision:
        return cls("deny", reason, detail=dict(detail))


# --------------------------------------------------------------------------- #
# Gate                                                                        #
# --------------------------------------------------------------------------- #


class SafetyGate:
    """Evaluate proposed tool calls against the configured policy.

    Parameters
    ----------
    config:
        The validated :class:`~friday.config.SafetyConfig`. The gate
        only reads — config changes mid-run aren't supported in Phase 1.
    registry:
        Where to look up the proposed skill's metadata. Used only to
        read the ``destructive`` flag and to detect unknown skills.
    """

    def __init__(self, config: SafetyConfig, registry: SkillRegistry) -> None:
        self._config = config
        self._registry = registry
        # Pre-lower-case the allowlists once so per-call matching stays
        # O(len(list)) without re-allocating strings every invocation.
        self._apps_lower: frozenset[str] = frozenset(
            a.strip().lower() for a in config.app_allowlist if a.strip()
        )
        self._shells: frozenset[str] = frozenset(
            s.strip() for s in config.shell_allowlist if s.strip()
        )

    # ----- skill invocations ------------------------------------------------

    def evaluate(
        self,
        skill_name: str,
        arguments: dict[str, Any],
    ) -> SafetyDecision:
        """Decide whether a proposed tool call may execute.

        Side effect: the decision is written to the audit log via
        :func:`audit`. The caller doesn't have to remember to log —
        and skipping it would leave a gap in the security trail.
        """
        decision = self._evaluate_inner(skill_name, arguments)
        self._record(skill_name, arguments, decision)
        return decision

    def _evaluate_inner(
        self,
        skill_name: str,
        arguments: dict[str, Any],
    ) -> SafetyDecision:
        sk = self._registry.get(skill_name)
        if sk is None:
            return SafetyDecision.deny(
                f"unknown skill: {skill_name!r}",
                skill=skill_name,
            )

        # App allowlist gate — comes before destructive/dry-run because
        # an off-allowlist app is interesting even in dry-run (we want
        # the LLM to be told that it picked something we don't allow,
        # rather than silently logging a dry-run line).
        if skill_name in _APP_GATED_SKILLS:
            app = str(arguments.get("name", "")).strip().lower()
            if not app:
                return SafetyDecision.deny(
                    "open_app called without a name",
                    skill=skill_name,
                )
            if app not in self._apps_lower:
                return SafetyDecision.needs_confirmation(
                    f"app {app!r} is not in app_allowlist",
                    skill=skill_name,
                    app=app,
                    allowlist=sorted(self._apps_lower),
                )

        # Destructive flag — outside dry-run, destructive always needs
        # a human in the loop. Inside dry-run, destructive flows through
        # to the dry-run branch below (audit logs intent, nothing runs).
        if sk.destructive and not self._config.dry_run:
            return SafetyDecision.needs_confirmation(
                f"skill {skill_name!r} is marked destructive",
                skill=skill_name,
            )

        # Dry-run gate — last so the audit log shows the most specific
        # reason for not running ("destructive + dry_run" → "dry_run",
        # which is correct: the destructive case in dry-run is exactly
        # what dry-run is for).
        if self._config.dry_run:
            return SafetyDecision.dry_run()

        return SafetyDecision.allow()

    # ----- shell commands ---------------------------------------------------

    def evaluate_shell(self, command: str) -> SafetyDecision:
        """Decide whether a shell invocation may execute.

        None of the Phase 1 builtins shell out, but the surface exists
        so the chunk 6 executor (and the Phase 3 code-exec sandbox) can
        route every ``subprocess.Popen`` proposal through here without
        a separate policy.
        """
        cmd = command.strip()
        if not cmd:
            decision = SafetyDecision.deny("empty shell command")
        else:
            # Allowlist matches the head of the command — `echo hello`
            # passes if `echo` is allowed.
            head = cmd.split(None, 1)[0]
            if head not in self._shells:
                decision = SafetyDecision.needs_confirmation(
                    f"shell {head!r} is not in shell_allowlist",
                    command=cmd,
                    allowlist=sorted(self._shells),
                )
            elif self._config.dry_run:
                decision = SafetyDecision.dry_run()
            else:
                decision = SafetyDecision.allow()

        # Filter ``command`` out of detail so the kwargs unpack below
        # doesn't collide with the top-level field of the same name.
        audit(
            "shell_invoke",
            command=cmd,
            decision=decision.kind,
            reason=decision.reason,
            **{k: v for k, v in decision.detail.items() if k != "command"},
        )
        return decision

    # ----- audit ------------------------------------------------------------

    def _record(
        self,
        skill_name: str,
        arguments: dict[str, Any],
        decision: SafetyDecision,
    ) -> None:
        audit(
            "skill_invoke",
            skill=skill_name,
            arguments=arguments,
            decision=decision.kind,
            reason=decision.reason,
            **{k: v for k, v in decision.detail.items() if k != "skill"},
        )
        # Mirror to the console at INFO so a developer running the loop
        # sees the gate's reasoning without having to tail the file.
        log.info(
            "safety: %s %s%s%s",
            decision.kind,
            skill_name,
            f" — {decision.reason}" if decision.reason else "",
            f"  args={arguments}" if arguments else "",
        )

    def record_execution(
        self,
        skill_name: str,
        arguments: dict[str, Any],
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Audit the outcome of an actually-executed skill.

        The executor calls this after running the skill so the audit
        log carries both the policy decision (from :meth:`evaluate`)
        and the realized outcome (from here). Keeping them as separate
        events makes it obvious whether a failure was policy-side or
        runtime-side.
        """
        audit(
            "skill_executed" if error is None else "skill_failed",
            skill=skill_name,
            arguments=arguments,
            result=None if error is not None else result,
            error=error,
        )


__all__ = ["DecisionKind", "SafetyDecision", "SafetyGate"]
