"""Tests for SkillRegistry behavior and the five builtin skills.

The schema-generation contract is exercised in ``test_skill_schema.py``.
This file focuses on registry-level concerns (dispatch, duplicates,
``tool_schemas`` output) and on the metadata of the builtins —
*not* their side effects, which Phase 1 still runs under dry-run.
"""

from __future__ import annotations

import pytest

from friday.skills.registry import SkillRegistry, skill


def test_dispatch_calls_through_with_kwargs() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def echo(text: str) -> str:
        """E."""
        return text.upper()

    assert reg.dispatch("echo", {"text": "hi"}) == "HI"


def test_dispatch_unknown_skill_raises_keyerror() -> None:
    reg = SkillRegistry()
    with pytest.raises(KeyError, match="unknown skill"):
        reg.dispatch("nope", {})


def test_duplicate_registration_rejected() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def dup(x: str) -> str:
        """X."""
        return x

    with pytest.raises(ValueError, match="already registered"):

        @skill(registry=reg, name="dup")
        def dup2(x: str) -> str:
            """Y."""
            return x


def test_names_are_sorted() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def zebra(x: str) -> str:
        """Z."""
        return x

    @skill(registry=reg)
    def alpha(x: str) -> str:
        """A."""
        return x

    assert reg.names() == ["alpha", "zebra"]


def test_tool_schemas_returns_one_per_skill() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def a(x: str) -> str:
        """A."""
        return x

    @skill(registry=reg)
    def b(y: int) -> str:
        """B."""
        return str(y)

    schemas = reg.tool_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"a", "b"}


def test_contains_only_matches_strings() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def s(x: str) -> str:
        """S."""
        return x

    assert "s" in reg
    assert "missing" not in reg
    assert 123 not in reg  # non-string keys never match


# --------------------------------------------------------------------------- #
# Builtins — metadata only (side effects are dry-run gated)                   #
# --------------------------------------------------------------------------- #


def test_builtins_register_on_package_import() -> None:
    # Importing the package triggers ``friday.skills.builtin`` import,
    # which decorates the five skills into ``default_registry``.
    from friday.skills import default_registry

    expected = {
        "open_app",
        "web_search",
        "media_control",
        "system_info",
        "type_text",
    }
    assert expected.issubset(set(default_registry.names()))


def test_builtin_open_app_schema() -> None:
    from friday.skills.builtin import open_app

    sk = open_app.skill  # type: ignore[attr-defined]
    assert sk.name == "open_app"
    assert sk.destructive is False
    assert sk.parameters_schema["required"] == ["name"]
    assert sk.parameters_schema["properties"]["name"]["type"] == "string"


def test_builtin_media_control_enum() -> None:
    from friday.skills.builtin import media_control

    sk = media_control.skill  # type: ignore[attr-defined]
    prop = sk.parameters_schema["properties"]["action"]
    assert prop["type"] == "string"
    assert prop["enum"] == [
        "play_pause",
        "next",
        "previous",
        "volume_up",
        "volume_down",
    ]


def test_builtin_system_info_enum() -> None:
    from friday.skills.builtin import system_info

    sk = system_info.skill  # type: ignore[attr-defined]
    prop = sk.parameters_schema["properties"]["metric"]
    assert prop["enum"] == ["battery", "time", "cpu"]


def test_builtin_type_text_is_marked_destructive() -> None:
    from friday.skills.builtin import type_text

    sk = type_text.skill  # type: ignore[attr-defined]
    assert sk.destructive is True


def test_system_info_time_runs_without_extras() -> None:
    # ``time`` is the only branch that doesn't need psutil — it should
    # work cleanly even when the skills extras aren't installed.
    from friday.skills.builtin import system_info

    result = system_info(metric="time")
    # HH:MM:SS — eight chars including the colons.
    assert len(result) == 8
    assert result.count(":") == 2
