"""Morning (or any-time) briefing skill.

Combines time-of-day greeting, weather, pending reminders, and
a random stored memory into one spoken summary — JARVIS-style
situational awareness on demand.

Requires: weather skill (no extra key), timers skill, memory skill.
All three are Phase 3 and always available.
"""

from __future__ import annotations

import datetime
import logging

from friday.skills.registry import skill

log = logging.getLogger(__name__)


@skill(description="Give a full status briefing: greeting, weather, reminders, and a memory.")
def morning_briefing(location: str = "") -> str:
    """Deliver a status briefing combining weather, reminders, and memory.

    Greets the user by time of day, reports local weather, lists any
    pending reminders, and surfaces a random stored memory as a
    conversation starter.

    Parameters
    ----------
    location:
        City for the weather report. Leave empty to auto-detect.
    """
    now = datetime.datetime.now()
    hour = now.hour

    if hour < 12:
        greeting = "Good morning, boss"
    elif hour < 17:
        greeting = "Good afternoon, boss"
    else:
        greeting = "Good evening, boss"

    time_str = now.strftime("It's %I:%M %p on %A, %B %d.")

    parts = [f"{greeting}. {time_str}"]

    # Weather
    try:
        from friday.skills.weather import get_weather
        weather_str = get_weather(location)
        parts.append(f"Weather: {weather_str}")
    except Exception as exc:
        log.debug("briefing: weather fetch failed: %s", exc)
        parts.append("Weather unavailable right now.")

    # Pending reminders
    try:
        from friday.skills.timers import list_reminders
        reminders_str = list_reminders()
        if "No pending" not in reminders_str:
            parts.append(f"Reminders: {reminders_str}")
        else:
            parts.append("No pending reminders.")
    except Exception as exc:
        log.debug("briefing: reminders check failed: %s", exc)

    # A memory snippet
    try:
        from friday.skills.memory import list_memories
        mem_str = list_memories()
        if "empty" not in mem_str.lower():
            # Pull just the first memory for the briefing.
            lines = [l for l in mem_str.splitlines() if l.strip() and l[0].isdigit()]
            if lines:
                first = lines[0].split(".", 1)[-1].strip()
                parts.append(f"From memory: {first}")
    except Exception as exc:
        log.debug("briefing: memory check failed: %s", exc)

    return " | ".join(parts)


__all__ = ["morning_briefing"]
