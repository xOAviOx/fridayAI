"""Built-in FRIDAY skills.

Phase 1 shipped five starter skills (open_app, web_search,
media_control, system_info, type_text).  Phase 2 adds six more:

* ``set_volume`` — set system output volume 0-100.
* ``take_screenshot`` — save a screenshot to disk.
* ``get_clipboard`` — read the current clipboard text.
* ``set_clipboard`` — write text to the clipboard (destructive).
* ``list_running_apps`` — list visible running processes.
* ``close_app`` — terminate an app by name (destructive).

Phase 2 also adds four file-system skills:

* ``read_file``     — return the text contents of a file.
* ``write_file``    — create or overwrite a file with text (destructive).
* ``delete_file``   — permanently delete a single file (destructive).
* ``create_folder`` — make a directory (and any missing parents).
* ``delete_folder`` — permanently remove a directory tree (destructive).

All skills are lazy-import-clean: the module can be imported without any
optional extra installed.  Only the *called* skill body pays the import
cost.

Safety note: the safety layer is what actually gates these in dry-run
mode.  ``destructive=True`` is the marker it reads.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Literal
import urllib.request
from urllib.parse import quote_plus

from friday.skills.registry import skill

log = logging.getLogger(__name__)


@skill(description="Open an application by name.")
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


@skill(description="Open Google search results in the browser.")
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
    url = f"https://www.google.com/search?q={quote_plus(q)}"
    webbrowser.open_new_tab(url)
    return f"opened browser to results for: {q}"


@skill(description="Search YouTube and open results in the browser.")
def youtube_search(query: str) -> str:
    """Open the default browser to YouTube search results for ``query``.

    Parameters
    ----------
    query:
        Search terms, e.g. ``"lo-fi beats"``, ``"how to make pasta"``.
    """
    q = query.strip()
    if not q:
        raise ValueError("youtube_search requires a non-empty query")
    url = f"https://www.youtube.com/results?search_query={quote_plus(q)}"
    webbrowser.open_new_tab(url)
    return f"opened YouTube search for: {q}"


@skill(description="Send a media key: play_pause, next, previous, volume_up, volume_down.")
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


@skill(description="Return current time, battery %, or CPU load.")
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


@skill(destructive=True, description="Type text at the current cursor position.")
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


@skill(description="Set system output volume 0–100.")
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


@skill(description="Save a screenshot to disk.")
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


@skill(description="Read and return the current clipboard text.")
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


@skill(destructive=True, description="Replace the clipboard with text.")
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


@skill(description="List names of currently running apps (up to 30).")
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


@skill(destructive=True, description="Terminate an application by name.")
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


# --------------------------------------------------------------------------- #
# File-system skills                                                          #
# --------------------------------------------------------------------------- #

# Maximum bytes read_file will return — keeps the LLM context manageable.
_READ_FILE_MAX_BYTES = 32_768  # 32 KB


@skill(description="Return the text content of a file (max 32 KB).")
def read_file(path: str) -> str:
    """Return the text content of a file.

    Reads the file at *path* as UTF-8 text and returns its content.
    Files larger than 32 KB are truncated with a notice so the LLM
    context doesn't blow up.

    Parameters
    ----------
    path:
        Absolute or relative path to the file.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{p} is a directory — use list_folder instead")

    size = p.stat().st_size
    raw = p.read_bytes()[:_READ_FILE_MAX_BYTES]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    suffix = f"\n[truncated — showing first {_READ_FILE_MAX_BYTES} of {size} bytes]" if size > _READ_FILE_MAX_BYTES else ""
    return text + suffix


@skill(destructive=True, description="Create or overwrite a file with text content.")
def write_file(path: str, content: str) -> str:
    """Write *content* to a file, creating it (and any parent directories) if needed.

    If the file already exists it is **overwritten** without warning —
    that is why this skill is marked destructive.  The caller should use
    ``read_file`` first if they want to confirm what will be replaced.

    Parameters
    ----------
    path:
        Absolute or relative path to the target file.
    content:
        UTF-8 text to write.  May be empty (creates an empty file).
    """
    p = Path(path).expanduser()
    if p.is_dir():
        raise IsADirectoryError(f"{p} is a directory, not a file")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"wrote {len(content)} chars ({lines} lines) to {p}"


@skill(destructive=True, description="Alias for write_file — create a new file with text content.")
def create_file(path: str, content: str) -> str:
    """Alias for write_file — many models prefer this name."""
    return write_file(path=path, content=content)


@skill(destructive=True, description="Permanently delete a file.")
def delete_file(path: str) -> str:
    """Permanently delete a single file.

    **This cannot be undone** — the file is not moved to Trash; it is
    removed from the filesystem directly.

    Parameters
    ----------
    path:
        Absolute or relative path to the file to delete.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{p} is a directory — use delete_folder instead")

    p.unlink()
    return f"deleted file: {p}"


@skill(description="Create a directory and any missing parent directories.")
def create_folder(path: str) -> str:
    """Create a directory (and any missing parents).

    Safe to call if the directory already exists — it returns success
    without touching the existing directory.  Not marked destructive
    because it never removes or overwrites anything.

    Parameters
    ----------
    path:
        Absolute or relative path of the directory to create.
    """
    p = Path(path).expanduser()
    if p.exists() and not p.is_dir():
        raise FileExistsError(f"{p} already exists and is not a directory")

    p.mkdir(parents=True, exist_ok=True)
    return f"created folder: {p}"


@skill(destructive=True, description="Permanently delete a directory and all its contents.")
def delete_folder(path: str) -> str:
    """Permanently delete a directory and everything inside it.

    **This cannot be undone** — the entire directory tree is removed
    from the filesystem directly, not moved to Trash.  Use with care.

    Parameters
    ----------
    path:
        Absolute or relative path to the directory to remove.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such directory: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is a file — use delete_file instead")

    # Safety guard: refuse to delete the home directory or filesystem root.
    home = Path.home()
    try:
        p.resolve().relative_to(home.resolve())
        inside_home = True
    except ValueError:
        inside_home = False

    if p.resolve() in (home.resolve(), Path("/"), Path("C:\\")):
        raise ValueError(f"refusing to delete protected path: {p}")

    shutil.rmtree(p)
    return f"deleted folder and all contents: {p}"


# --------------------------------------------------------------------------- #
# Expanded system metrics                                                      #
# --------------------------------------------------------------------------- #


@skill(description="Return RAM usage: total, used, and free memory.")
def get_ram_usage() -> str:
    """Return current RAM usage in human-readable form.

    Reports total installed RAM, how much is currently used, and
    how much is free/available — plus the usage percentage.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "get_ram_usage requires psutil. Install skills extras:\n"
            "    uv sync --extra skills"
        ) from exc

    vm = psutil.virtual_memory()

    def _fmt(b: int) -> str:
        if b >= 1_073_741_824:
            return f"{b / 1_073_741_824:.1f} GB"
        return f"{b / 1_048_576:.0f} MB"

    return (
        f"RAM: {_fmt(vm.used)} used / {_fmt(vm.total)} total "
        f"({vm.percent:.0f}% used, {_fmt(vm.available)} free)"
    )


@skill(description="Return disk usage for the main drive.")
def get_disk_usage() -> str:
    """Return disk space usage for the root / main drive.

    Reports total capacity, used space, free space, and usage
    percentage for the filesystem mounted at ``/`` (or ``C:\\``
    on Windows).
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "get_disk_usage requires psutil. Install skills extras:\n"
            "    uv sync --extra skills"
        ) from exc

    mount = "C:\\" if sys.platform == "win32" else "/"
    disk = psutil.disk_usage(mount)

    def _fmt(b: int) -> str:
        if b >= 1_099_511_627_776:
            return f"{b / 1_099_511_627_776:.1f} TB"
        return f"{b / 1_073_741_824:.1f} GB"

    return (
        f"Disk ({mount}): {_fmt(disk.used)} used / {_fmt(disk.total)} total "
        f"({disk.percent:.0f}% used, {_fmt(disk.free)} free)"
    )


@skill(description="Return the machine's local and public IP addresses.")
def get_ip_address() -> str:
    """Return local and public IP addresses.

    Local IP is read from the default network interface. Public IP is
    fetched from ``https://api.ipify.org`` (requires internet access).
    """
    import socket  # stdlib — always available

    # Local IP via a dummy UDP connection (never actually sends data).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "unknown"

    # Public IP via ipify.
    try:
        req = urllib.request.Request(
            "https://api.ipify.org",
            headers={"User-Agent": "FRIDAY-AI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            public_ip = resp.read().decode("utf-8").strip()
    except Exception:
        public_ip = "unavailable (no internet?)"

    return f"Local IP: {local_ip} | Public IP: {public_ip}"


__all__ = [
    "close_app",
    "create_folder",
    "delete_file",
    "delete_folder",
    "get_clipboard",
    "get_disk_usage",
    "get_ip_address",
    "get_ram_usage",
    "list_running_apps",
    "media_control",
    "open_app",
    "read_file",
    "set_clipboard",
    "set_volume",
    "system_info",
    "take_screenshot",
    "type_text",
    "web_search",
    "write_file",
    "youtube_search",
]
