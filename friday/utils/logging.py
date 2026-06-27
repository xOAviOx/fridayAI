"""Logging setup.

Two streams:

* The root logger writes human-readable lines to stdout. Use this for
  everything: capture, transcription, LLM calls, errors.
* A separate `friday.audit` logger writes one line per *executed* action
  to ``audit_path``. Phase 3's safety layer will be the main caller —
  but it's wired up here so Phase 1 can already log dry-run intents.

The audit logger does not propagate to the root, so audit lines stay out
of the noisy console stream.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", audit_path: str | Path | None = None) -> None:
    """Configure the root logger and the dedicated audit logger.

    Safe to call more than once — existing handlers are removed first so
    tests and reloads don't double up output.
    """
    root = logging.getLogger()
    root.setLevel(_parse_level(level))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT))
    root.addHandler(console)

    if audit_path is not None:
        _setup_audit_logger(Path(audit_path))


def audit(event: str, **fields: Any) -> None:
    """Append one structured line to the audit log.

    Use this for every action that is (or would be) executed on the user's
    machine — even in dry-run mode. The line is JSON so it stays grep-able
    as the action set grows.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
    }
    payload.update(fields)
    logging.getLogger("friday.audit").info(json.dumps(payload, default=str))


def _setup_audit_logger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audit_logger = logging.getLogger("friday.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # don't double-print to console

    for handler in list(audit_logger.handlers):
        audit_logger.removeHandler(handler)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    # The audit format is just the timestamped JSON payload — the JSON
    # already carries the event name and fields.
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(file_handler)


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, level.upper())
    except AttributeError as exc:
        raise ValueError(f"Unknown log level: {level!r}") from exc
