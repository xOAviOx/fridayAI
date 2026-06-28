"""Persistent memory skills — FRIDAY remembers facts across sessions.

Memories are stored as JSON lines in ``~/.friday/memories.json``.
Each entry is a dict with ``fact``, ``timestamp``, and optional ``topic``.

Skills
------
``remember``      — store a new fact.
``recall``        — search memories by topic keyword.
``list_memories`` — dump all stored facts.
``forget``        — delete a matching memory.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from friday.skills.registry import skill

log = logging.getLogger(__name__)

_MEMORY_DIR = Path.home() / ".friday"
_MEMORY_FILE = _MEMORY_DIR / "memories.json"
_lock = threading.Lock()

# In-process cache so reads don't always hit disk.
_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is not None:
        return _cache
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not _MEMORY_FILE.exists():
        _cache = []
        return _cache
    try:
        data = json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
        _cache = data if isinstance(data, list) else []
    except Exception:
        log.warning("memory file corrupted — starting fresh")
        _cache = []
    return _cache


def _save(memories: list[dict]) -> None:
    global _cache
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_FILE.write_text(
        json.dumps(memories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _cache = memories


# --------------------------------------------------------------------------- #
# Skills                                                                      #
# --------------------------------------------------------------------------- #


@skill(description="Remember a fact for future conversations.")
def remember(fact: str) -> str:
    """Store a fact in FRIDAY's persistent memory.

    Parameters
    ----------
    fact:
        Any piece of information to remember, e.g.
        ``"My standup is at 10 AM every weekday"``,
        ``"Avi's birthday is March 15"``,
        ``"Prefer dark mode in all apps"``.
    """
    fact = fact.strip()
    if not fact:
        raise ValueError("remember requires a non-empty fact")

    entry = {
        "fact": fact,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with _lock:
        memories = _load()
        # Avoid exact duplicates.
        if any(m["fact"].lower() == fact.lower() for m in memories):
            return f"Already know that: {fact!r}"
        memories.append(entry)
        _save(memories)

    return f"Got it, I'll remember: {fact}"


@skill(description="Search FRIDAY's memory for facts about a topic.")
def recall(topic: str = "") -> str:
    """Search stored memories for facts related to *topic*.

    Returns all matching facts as a numbered list.  Pass an empty
    string (or omit) to get every stored memory.

    Parameters
    ----------
    topic:
        Keyword(s) to search for, e.g. ``"birthday"``, ``"standup"``.
        Case-insensitive. Leave empty to return all memories.
    """
    with _lock:
        memories = _load()

    if not memories:
        return "No memories stored yet. Use 'remember' to add some."

    kw = topic.strip().lower()
    if kw:
        matched = [m for m in memories if kw in m["fact"].lower()]
    else:
        matched = memories

    if not matched:
        return f"Nothing in memory matching '{topic}'."

    lines = [f"{i + 1}. {m['fact']}" for i, m in enumerate(matched)]
    header = f"Found {len(matched)} memor{'y' if len(matched) == 1 else 'ies'}:"
    return header + "\n" + "\n".join(lines)


@skill(description="List everything FRIDAY remembers.")
def list_memories() -> str:
    """Return all stored memories as a numbered list."""
    with _lock:
        memories = _load()

    if not memories:
        return "Memory is empty."

    lines = [f"{i + 1}. {m['fact']}" for i, m in enumerate(memories)]
    return f"{len(memories)} stored memor{'y' if len(memories) == 1 else 'ies'}:\n" + "\n".join(lines)


@skill(destructive=True, description="Delete a memory that matches the given text.")
def forget(fact: str) -> str:
    """Remove a stored memory that contains *fact* as a substring.

    Deletes the first memory whose text contains *fact* (case-insensitive).
    Use :func:`list_memories` first to see exact wording.

    Parameters
    ----------
    fact:
        Substring of the fact to remove.
    """
    fact = fact.strip()
    if not fact:
        raise ValueError("forget requires a non-empty fact description")

    kw = fact.lower()
    with _lock:
        memories = _load()
        original_count = len(memories)
        kept = [m for m in memories if kw not in m["fact"].lower()]
        removed = original_count - len(kept)
        if removed:
            _save(kept)

    if removed == 0:
        return f"Nothing in memory matches '{fact}'."
    return f"Forgot {removed} memor{'y' if removed == 1 else 'ies'} matching '{fact}'."


__all__ = ["forget", "list_memories", "recall", "remember"]
