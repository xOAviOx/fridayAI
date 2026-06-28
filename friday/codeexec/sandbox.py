"""Code execution skills — Phase 5.

Two skills gated behind ``safety.enable_code_exec: true``:

``run_python``
    Execute arbitrary Python code in a fresh subprocess.  Captures
    stdout + stderr and returns them as a string so the LLM can reason
    about the output.  Hard 10-second timeout; the process is killed if
    it exceeds it.  Useful for: maths, data manipulation, generating
    files programmatically, calling any Python library.

``run_shell``
    Run a shell command.  The command head must appear in
    ``safety.shell_allowlist`` in config.yaml — add whatever you need.
    Hard 15-second timeout.  Useful for: git status, directory listing,
    running scripts, npm / pip installs, process inspection.

Both skills are also marked ``destructive=True`` so the safety layer
requires ``skip_confirm_destructive: true`` (or a confirmation prompt)
when dry_run is off — an extra opt-in on top of enable_code_exec.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from friday.skills.registry import skill

log = logging.getLogger(__name__)

_HOME = Path.home()

# --------------------------------------------------------------------------- #
# Python sandbox                                                               #
# --------------------------------------------------------------------------- #

@skill(
    destructive=True,
    description="Execute Python code in a subprocess and return stdout + stderr.",
)
def run_python(code: str, timeout_s: int = 10) -> str:
    """Run Python code and return its output.

    Executes *code* using the same Python interpreter that is running
    FRIDAY, in a fresh child process.  Both stdout and stderr are
    captured and returned so the LLM can read the result.

    If the process takes longer than *timeout_s* seconds it is killed
    and a ``[timeout]`` marker is returned.

    Useful for: maths and unit conversions, manipulating files with code,
    calling any installed Python library, generating content
    programmatically, running quick data transforms.

    Parameters
    ----------
    code:
        Valid Python source code.  Multi-line strings work — the code
        is passed via a temporary stdin pipe, not a ``-c`` flag, so
        there is no restriction on quotes or whitespace.
    timeout_s:
        Maximum execution time in seconds (default 10).  Raise this
        for long-running scripts; lower it for quick one-liners.
    """
    code = code.strip()
    if not code:
        raise ValueError("run_python: code must not be empty")

    # Inherit the active venv so installed packages are available.
    env = {**os.environ}

    try:
        proc = subprocess.run(
            [sys.executable, "-"],   # read code from stdin
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(_HOME),          # safe, user-owned working dir
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] Python code timed out after {timeout_s}s"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        label = "[stderr]" if proc.returncode == 0 else "[error]"
        parts.append(f"{label} {stderr}")
    if proc.returncode != 0 and not stderr:
        parts.append(f"[exit {proc.returncode}]")

    return "\n".join(parts) if parts else "(no output)"


# --------------------------------------------------------------------------- #
# Shell runner                                                                 #
# --------------------------------------------------------------------------- #

@skill(
    destructive=True,
    description=(
        "Run a shell command from the allowlist and return its output. "
        "Add commands to safety.shell_allowlist in config.yaml."
    ),
)
def run_shell(command: str, timeout_s: int = 15) -> str:
    """Run a shell command and return its output.

    The first word of *command* must be present in
    ``safety.shell_allowlist`` in ``config.yaml`` — the safety gate
    enforces this before the skill body even runs.  Add any command you
    trust to that list.

    Captures stdout + stderr.  Hard *timeout_s* second timeout.

    Useful for: ``git status / log / diff``, ``ls / find``, running
    scripts, ``npm run``, ``pip list``, process inspection with ``ps``.

    Parameters
    ----------
    command:
        Full shell command string, e.g. ``"git status"``,
        ``"ls -la ~/Desktop"``, ``"pip list | grep torch"``.
    timeout_s:
        Maximum execution time in seconds (default 15).
    """
    command = command.strip()
    if not command:
        raise ValueError("run_shell: command must not be empty")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(_HOME),
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] shell command timed out after {timeout_s}s"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        label = "[stderr]" if proc.returncode == 0 else "[error]"
        parts.append(f"{label} {stderr}")
    if proc.returncode != 0 and not stderr and not stdout:
        parts.append(f"[exit {proc.returncode}]")

    result = "\n".join(parts) if parts else "(no output)"
    # Cap output so we don't blow up the LLM context.
    if len(result) > 4_000:
        result = result[:4_000] + "\n[... truncated]"
    return result


__all__ = ["run_python", "run_shell"]
