"""Reminders and timers — FRIDAY speaks when the time comes.

Architecture
------------
A single background ``_MonitorThread`` wakes every second, checks
pending reminders, and fires any that are due by calling a
``speak_callback`` registered by the agent loop.

The loop calls :func:`set_speak_callback` once on startup — before
any skill can fire — so the callback is always ready.

Skills
------
``set_reminder``    — speak a message after N minutes.
``set_timer``       — simple countdown with a label.
``list_reminders``  — see what's pending.
``cancel_reminder`` — cancel by ID.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from friday.skills.registry import skill

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Internal state                                                               #
# --------------------------------------------------------------------------- #

_speak_callback: Callable[[str], None] | None = None
_lock = threading.Lock()
_reminders: dict[str, "_Reminder"] = {}
_monitor_started = False


@dataclass
class _Reminder:
    id: str
    message: str
    fire_at: float        # Unix timestamp
    label: str = ""
    fired: bool = False

    def due(self) -> bool:
        return not self.fired and time.time() >= self.fire_at

    def summary(self) -> str:
        due_in = self.fire_at - time.time()
        if due_in <= 0:
            return f"[{self.id[:8]}] {self.message} — firing now"
        minutes = int(due_in // 60)
        seconds = int(due_in % 60)
        if minutes > 0:
            return f"[{self.id[:8]}] {self.message} — in {minutes}m {seconds}s"
        return f"[{self.id[:8]}] {self.message} — in {seconds}s"


# --------------------------------------------------------------------------- #
# Callback registration (called by the agent loop)                            #
# --------------------------------------------------------------------------- #


def set_speak_callback(fn: Callable[[str], None]) -> None:
    """Register the TTS callback the monitor uses to fire reminders.

    Called once by :class:`~friday.agent.loop.AgentLoop` after the TTS
    system is ready.  ``fn`` must be safe to call from a background thread.
    """
    global _speak_callback
    _speak_callback = fn
    _ensure_monitor_running()


# --------------------------------------------------------------------------- #
# Background monitor thread                                                   #
# --------------------------------------------------------------------------- #


def _ensure_monitor_running() -> None:
    global _monitor_started
    if _monitor_started:
        return
    _monitor_started = True
    t = threading.Thread(target=_monitor_loop, daemon=True, name="friday-timers")
    t.start()
    log.debug("timer monitor thread started")


def _monitor_loop() -> None:
    while True:
        time.sleep(1)
        with _lock:
            due = [r for r in _reminders.values() if r.due()]
        for reminder in due:
            with _lock:
                reminder.fired = True
            msg = reminder.message
            log.info("timer fired: %s", msg)
            cb = _speak_callback
            if cb is not None:
                # Fire in its own thread so we don't block the monitor.
                threading.Thread(
                    target=cb,
                    args=(f"Hey boss — {msg}",),
                    daemon=True,
                    name=f"friday-reminder-{reminder.id[:8]}",
                ).start()
            else:
                log.warning("reminder fired but no speak callback registered: %s", msg)


# --------------------------------------------------------------------------- #
# Skills                                                                      #
# --------------------------------------------------------------------------- #


@skill(description="Set a spoken reminder after N minutes.")
def set_reminder(message: str, in_minutes: float) -> str:
    """Schedule a spoken reminder.

    FRIDAY will speak *message* after *in_minutes* minutes, even in
    the middle of another task.

    Parameters
    ----------
    message:
        What FRIDAY should say when the reminder fires,
        e.g. ``"take your medicine"``, ``"meeting starts now"``.
    in_minutes:
        Delay in minutes (can be fractional, e.g. 0.5 for 30 seconds).
        Must be positive.
    """
    message = message.strip()
    if not message:
        raise ValueError("set_reminder requires a non-empty message")
    if in_minutes <= 0:
        raise ValueError("in_minutes must be positive")

    _ensure_monitor_running()

    rid = str(uuid.uuid4())
    fire_at = time.time() + in_minutes * 60
    reminder = _Reminder(id=rid, message=message, fire_at=fire_at)

    with _lock:
        _reminders[rid] = reminder

    mins = int(in_minutes)
    secs = int((in_minutes - mins) * 60)
    if mins > 0 and secs > 0:
        when = f"{mins}m {secs}s"
    elif mins > 0:
        when = f"{mins} minute{'s' if mins != 1 else ''}"
    else:
        when = f"{secs} second{'s' if secs != 1 else ''}"

    return f"Reminder set: '{message}' in {when}. ID: {rid[:8]}"


@skill(description="Start a countdown timer with a label.")
def set_timer(minutes: float, label: str = "timer") -> str:
    """Start a countdown timer.

    Works exactly like :func:`set_reminder` but uses a generic
    "timer done" framing. Convenient for cooking, workouts, etc.

    Parameters
    ----------
    minutes:
        Timer duration in minutes (can be fractional).
    label:
        Short name for the timer, e.g. ``"pasta"``, ``"focus block"``.
    """
    label = label.strip() or "timer"
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    message = f"{label} is done"
    return set_reminder(message=message, in_minutes=minutes)


@skill(description="List all pending (not yet fired) reminders.")
def list_reminders() -> str:
    """Return a summary of reminders that haven't fired yet."""
    with _lock:
        pending = [r for r in _reminders.values() if not r.fired]

    if not pending:
        return "No pending reminders."

    lines = [r.summary() for r in sorted(pending, key=lambda r: r.fire_at)]
    return f"{len(pending)} pending reminder{'s' if len(pending) != 1 else ''}:\n" + "\n".join(lines)


@skill(destructive=True, description="Cancel a pending reminder by its ID (first 8 chars work).")
def cancel_reminder(reminder_id: str) -> str:
    """Cancel a reminder before it fires.

    Parameters
    ----------
    reminder_id:
        The ID shown by :func:`list_reminders` or :func:`set_reminder`.
        You only need the first 8 characters.
    """
    rid = reminder_id.strip()
    if not rid:
        raise ValueError("cancel_reminder requires a reminder ID")

    with _lock:
        # Support full UUID or the 8-char prefix shown to the user.
        match = None
        for full_id, r in _reminders.items():
            if full_id == rid or full_id.startswith(rid):
                match = r
                break

        if match is None:
            return f"No reminder found with ID '{rid}'."

        if match.fired:
            return f"Reminder '{match.message}' already fired — nothing to cancel."

        match.fired = True  # Mark as fired (won't speak) rather than deleting.

    return f"Cancelled reminder: '{match.message}'."


__all__ = [
    "cancel_reminder",
    "list_reminders",
    "set_remind_callback",
    "set_reminder",
    "set_speak_callback",
    "set_timer",
]
