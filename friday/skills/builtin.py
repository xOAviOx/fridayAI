"""The five starter skills the brief calls out by name.

Each is intentionally small — the chunk's value is in the registry's
schema generation, not in the skill bodies themselves. The bodies do
the real OS work, but every dependency that isn't stdlib is
lazy-imported so importing this module is free.

Importing this module has the side effect of registering all five
skills in :data:`friday.skills.registry.default_registry`. That's how
``friday/skills/__init__.py`` populates the default registry on
package init.

Safety note: the safety layer (chunk 4) is what actually gates these
in dry-run mode and against the shell/app allowlists. By the time
control reaches a skill body the action has already been approved.
``destructive=True`` is the marker the safety layer reads.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import subprocess
import sys
import webbrowser
from typing import Literal
from urllib.parse import quote_plus

from friday.skills.registry import skill

log = logging.getLogger(__name__)


@skill
def open_app(name: str) -> str:
    """Open the named application on the user's machine.

    Parameters
    ----------
    name:
        Application name as the user would type into the Start menu /
        Spotlight / a shell, e.g. ``"spotify"``, ``"chrome"``,
        ``"notepad"``. Allowlist matching happens in the safety layer
        — by the time we get here, ``name`` is approved.
    """
    target = name.strip()
    if not target:
        raise ValueError("open_app requires a non-empty name")

    if sys.platform == "win32":
        # ``start ""`` lets Windows resolve the app via the Start menu
        # the same way the user would — handles Microsoft Store apps,
        # PATH lookups, and registered launchers without us having to
        # care which one applies.
        subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", target])
    else:
        # Linux: try the bare command first, fall back to xdg-open so
        # ``open_app("firefox")`` works whether the user's PATH knows it
        # directly or only through the desktop database.
        if shutil.which(target):
            subprocess.Popen([target])
        else:
            subprocess.Popen(["xdg-open", target])
    return f"launched {target}"


@skill
def web_search(query: str) -> str:
    """Open the default browser to search results for ``query``.

    Parameters
    ----------
    query:
        Free-text search query. Routed to DuckDuckGo because it needs
        no API key, no engine-specific schema, and respects the
        existing browser's default session (auth, extensions, etc.).
    """
    q = query.strip()
    if not q:
        raise ValueError("web_search requires a non-empty query")
    url = f"https://duckduckgo.com/?q={quote_plus(q)}"
    webbrowser.open_new_tab(url)
    return f"opened browser to results for: {q}"


@skill
def media_control(
    action: Literal[
        "play_pause", "next", "previous", "volume_up", "volume_down"
    ],
) -> str:
    """Send an OS-level media key.

    Parameters
    ----------
    action:
        ``play_pause`` toggles playback in the focused media app,
        ``next`` / ``previous`` skip tracks, ``volume_up`` /
        ``volume_down`` step the system volume. All five map to the
        same media keys a keyboard's media row would send, so any
        well-behaved media app (Spotify, browser HTML audio, system
        player) reacts.
    """
    try:
        import pyautogui  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "media_control requires pyautogui. Install the skills extras:\n"
            "    uv sync --extra skills\n"
            "or:\n"
            "    pip install -e '.[skills]'"
        ) from exc

    key_map = {
        "play_pause": "playpause",
        "next": "nexttrack",
        "previous": "prevtrack",
        "volume_up": "volumeup",
        "volume_down": "volumedown",
    }
    pyautogui.press(key_map[action])
    return f"sent media key: {action}"


@skill
def system_info(metric: Literal["battery", "time", "cpu"]) -> str:
    """Return a one-line snapshot of a system metric.

    Parameters
    ----------
    metric:
        ``time`` — current local wall-clock time (HH:MM:SS).
        ``battery`` — percent + charging status, or ``"no battery"``
        on a desktop.
        ``cpu`` — CPU load percentage sampled over one second.
    """
    if metric == "time":
        return datetime.datetime.now().strftime("%H:%M:%S")

    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "system_info battery/cpu require psutil. Install the skills extras:\n"
            "    uv sync --extra skills\n"
            "or:\n"
            "    pip install -e '.[skills]'"
        ) from exc

    if metric == "battery":
        bat = psutil.sensors_battery()
        if bat is None:
            return "no battery detected"
        plugged = " (charging)" if bat.power_plugged else ""
        return f"{bat.percent:.0f}%{plugged}"

    if metric == "cpu":
        return f"{psutil.cpu_percent(interval=1.0):.1f}%"

    # The Literal is exhaustive, but be explicit so a future addition
    # to the enum doesn't silently fall off the bottom.
    raise ValueError(f"unknown metric: {metric!r}")


@skill(destructive=True)
def type_text(text: str) -> str:
    """Type ``text`` at the current keyboard cursor position.

    Destructive because it modifies whatever document, field, or
    terminal is currently focused. The safety layer (chunk 4) is
    responsible for confirming this with the user before the executor
    calls through; ``destructive=True`` is the marker it reads.

    Parameters
    ----------
    text:
        Literal text to type. Newlines render as Enter, tabs as Tab.
        Clipboard is not touched.
    """
    if not text:
        raise ValueError("type_text requires non-empty text")
    try:
        import pyautogui  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "type_text requires pyautogui. Install the skills extras:\n"
            "    uv sync --extra skills\n"
            "or:\n"
            "    pip install -e '.[skills]'"
        ) from exc
    pyautogui.typewrite(text)
    return f"typed {len(text)} characters"


__all__ = [
    "media_control",
    "open_app",
    "system_info",
    "type_text",
    "web_search",
]
