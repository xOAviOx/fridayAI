"""FRIDAY entry point.

Phase 0: load + validate config, configure logging, print readiness, exit.
The actual event loop lands in Phase 1.

Exit codes
----------
0   normal exit
2   configuration error (bad yaml, missing required secret, etc.)
130 user interrupt (Ctrl+C)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from friday import __version__
from friday.config import Config, ConfigError, load_config
from friday.utils.logging import audit, setup_logging

log = logging.getLogger("friday")


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI args yet; the brief calls for `python -m friday.main`

    _enable_utf8_streams()

    root = Path.cwd()
    config_path = root / "config.yaml"
    env_path = root / ".env"

    try:
        config = load_config(config_path=config_path, env_path=env_path)
    except ConfigError as exc:
        # Logging isn't configured yet — go straight to stderr so the
        # user actually sees this.
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    setup_logging(level=config.logging.level, audit_path=config.logging.audit_path)
    _log_readiness(config)
    audit("boot", phase="0", version=__version__)

    log.info("Phase 0 scaffold: no event loop yet — exiting cleanly.")
    return 0


def _enable_utf8_streams() -> None:
    """Switch stdout/stderr to UTF-8 on Windows so unicode (— etc.) renders.

    On Linux/macOS stdout is already UTF-8 and ``reconfigure`` is a no-op.
    On older Pythons or non-TTY streams this can fail silently — that's
    fine, we just fall back to whatever the system default is.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - best effort
            pass


def _log_readiness(config: Config) -> None:
    """Print a one-screen summary of how FRIDAY is configured."""
    log.info("FRIDAY ready. (v%s)", __version__)
    log.info("  brain (LLM):  %s", config.providers.llm)
    log.info("  ears  (STT):  %s", config.providers.stt)
    log.info("  mouth (TTS):  %s", config.providers.tts)
    log.info(
        "  hotkeys:      push=%s  panic=%s",
        config.hotkeys.push_to_talk,
        config.hotkeys.panic,
    )
    log.info(
        "  safety:       dry_run=%s  code_exec=%s  computer_use=%s",
        config.safety.dry_run,
        config.safety.enable_code_exec,
        config.safety.enable_computer_use,
    )
    if not config.safety.dry_run:
        log.warning(
            "dry_run is OFF — actions will execute on this machine when "
            "Phase 1 lands. Flip it back in config.yaml while testing."
        )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print()  # newline after ^C
        sys.exit(130)
