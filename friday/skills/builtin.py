"""Built-in FRIDAY skills.

Phase 1 shipped five starter skills (open_app, web_search,
media_control, system_info, type_text).  Phase 2 adds six more:

* ``set_volume`` — set system output volume 0-100.
* ``take_screenshot`` — save a screenshot to disk.
* ``get_clipboard`` — read the current clipboard text.
* ``set_clipboard`` — write text to the clipboard (destructive).
* ``list_running_apps`` — list visible running processes.
* ``close_app`` — terminate an app by name (destructive).

All six are lazy-import-clean: the module can be imported without any
optional extra installed.  Only the *called* skill body pays the import
cost.

Safety note: the safety layer is what actually gates these in dry-run
mode.  ``destructive=True`` is the marker it reads.
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


@skill
def set_volume(level: int) -> str:
    """Set the system output volume.

    Parameters
    ----------
    level:
        Target volume 0–100. 0 = mute, 100 = maximum.
    """
    if not 0 <= level <= 100:
        raise ValueError(f"level must be 0-100, got {level}")

    if sys.platform == "darwin":
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            check=True,
            capture_output=True,
        )
    elif sys.platform == "win32":
        # Use PowerShell's WScript.Shell to set volume via the mixer.
        script = (
            f"$obj = New-Object -ComObject WScript.Shell; "
            f"$vol = [int]({level} * 65535 / 100); "
            f"(New-Object -ComObject WMPlayer.OCX.7).settings.volume = {level}"
        )
        # Simpler nircmd path if available; fall back to audio key press.
        if shutil.which("nircmd"):
            vol_val = int(level * 65535 / 100)
            subprocess.run(["nircmd", "sndvol", "set", str(vol_val)], check=True)
        else:
            subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True,
            )
    else:
        # Linux: amixer or pactl (PulseAudio / PipeWire)
        if shutil.which("amixer"):
            subprocess.run(
                ["amixer", "-q", "sset", "Master", f"{level}%"],
                check=True,
            )
        elif shutil.which("pactl"):
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=True,
            )
        else:
            raise RuntimeError("no volume control found (tried amixer, pactl)")

    return f"volume set to {level}%"


@skill
def take_screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot and save it to disk.

    Parameters
    ----------
    filename:
        Output file name (relative to the current working directory or
        an absolute path). Defaults to ``screenshot.png``.  The format
        is inferred from the extension (``png``, ``jpg``, ``bmp``).
    """
    try:
        import pyautogui  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "take_screenshot requires pyautogui. Install the skills extras:\n"
            "    uv sync --extra skills"
        ) from exc

    path = filename.strip() or "screenshot.png"
    img = pyautogui.screenshot()
    img.save(path)
    return f"screenshot saved to {path}"


@skill
def get_clipboard() -> str:
    """Return the current clipboard text.

    Reads from the system clipboard.  Returns an empty string when the
    clipboard is empty or contains non-text content.
    """
    if sys.platform == "darwin":
        result = subprocess.run(
            ["pbpaste"], capture_output=True, text=True, timeout=5
        )
        return result.stdout
    elif sys.platform == "win32":
        result = subprocess.run(
            ["powershell", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.rstrip("\r\n")
    else:
        # Linux: try xclip then xsel
        for cmd in [["xclip", "-selection", "clipboard", "-o"],
                    ["xsel", "--clipboard", "--output"]]:
            if shutil.which(cmd[0]):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                return result.stdout
        raise RuntimeError("no clipboard tool found (tried xclip, xsel)")


@skill(destructive=True)
def set_clipboard(text: str) -> str:
    """Write text to the system clipboard.

    Marked destructive because it silently overwrites whatever the user
    currently has on the clipboard.

    Parameters
    ----------
    text:
        The text to place on the clipboard.
    """
    if not text:
        raise ValueError("set_clipboard requires non-empty text")

    if sys.platform == "darwin":
        subprocess.run(
            ["pbcopy"], input=text.encode(), check=True, timeout=5
        )
    elif sys.platform == "win32":
        subprocess.run(
            ["powershell", "-Command", f"Set-Clipboard -Value '{text}'"],
            check=True, timeout=5,
        )
    else:
        for cmd, extra in [
            (["xclip", "-selection", "clipboard"], {}),
            (["xsel", "--clipboard", "--input"], {}),
        ]:
            if shutil.which(cmd[0]):
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
                break
        else:
            raise RuntimeError("no clipboard tool found (tried xclip, xsel)")

    preview = text[:40] + ("…" if len(text) > 40 else "")
    return f"clipboard set to: {preview!r}"


@skill
def list_running_apps() -> str:
    """Return a deduplicated list of visible running application names.

    Uses ``psutil`` to enumerate processes and filters out kernel
    threads and system daemons, returning only user-visible process
    names.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "list_running_apps requires psutil. Install the skills extras:\n"
            "    uv sync --extra skills"
        ) from exc

    names: set[str] = set()
    for proc in psutil.process_iter(["name"]):
        try:
            name = proc.info.get("name") or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # Strip common system noise — empty names, kernel helpers.
        name = name.strip()
        if name and not name.startswith("["):
            # Trim .exe suffix on Windows for readability.
            if name.lower().endswith(".exe"):
                name = name[:-4]
            names.add(name)

    if not names:
        return "no running apps found"
    sorted_names = sorted(names, key=str.lower)
    return ", ".join(sorted_names[:30])  # cap at 30 to avoid a wall of text


@skill(destructive=True)
def close_app(name: str) -> str:
    """Close (terminate) the named application.

    Marked destructive — it sends SIGTERM (or the OS equivalent) to
    all processes whose name matches ``name``.  Unsaved work in the
    target app will be lost.

    Parameters
    ----------
    name:
        Process name to kill, e.g. ``"spotify"``, ``"chrome"``.
        Matching is case-insensitive.  On Windows, ``.exe`` is added
        automatically if absent.
    """
    target = name.strip()
    if not target:
        raise ValueError("close_app requires a non-empty name")

    if sys.platform == "win32":
        exe = target if target.lower().endswith(".exe") else f"{target}.exe"
        result = subprocess.run(
            ["taskkill", "/IM", exe, "/F"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return f"closed {target}"
        return f"could not close {target}: {result.stderr.strip()}"

    elif sys.platform == "darwin":
        # Try AppleScript first (graceful quit), fall back to pkill.
        result = subprocess.run(
            ["osascript", "-e", f'quit app "{target}"'],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return f"asked {target} to quit"
        # Fall through to pkill for apps that don't respond to AppleScript.
        subprocess.run(["pkill", "-ix", target], capture_output=True)
        return f"sent terminate signal to {target}"

    else:
        # Linux: pkill by name.
        result = subprocess.run(
            ["pkill", "-x", target],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return f"terminated {target}"
        # returncode 1 means no matching process.
        return f"no process named {target!r} found"


__all__ = [
    "close_app",
    "get_clipboard",
    "list_running_apps",
    "media_control",
    "open_app",
    "set_clipboard",
    "set_volume",
    "system_info",
    "take_screenshot",
    "type_text",
    "web_search",
]
