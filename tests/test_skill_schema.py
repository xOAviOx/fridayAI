"""Tests for ``@skill`` schema generation.

These are the schema-generation tests the brief calls out by name —
the spec's bar for chunk 3 is "the LLM sees the schema we say it does
for any valid skill signature." Every supported type, every supported
edge case in docstring parsing, every shape of the OpenAI tool schema
gets one assertion here.

Each test builds a fresh ad-hoc function so we never depend on the
global ``default_registry`` state.
"""

from __future__ import annotations

from typing import Literal

import pytest

from friday.skills.registry import (
    SkillRegistry,
    build_skill,
    skill,
)


# --------------------------------------------------------------------------- #
# Basics                                                                      #
# --------------------------------------------------------------------------- #


def test_simple_string_param() -> None:
    def f(name: str) -> str:
        """Open something.

        Parameters
        ----------
        name:
            The thing to open.
        """
        return name

    built = build_skill(f)
    assert built.name == "f"
    assert built.description == "Open something."

    props = built.parameters_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["name"]["description"] == "The thing to open."
    assert built.parameters_schema["required"] == ["name"]
    assert built.parameters_schema["additionalProperties"] is False


def test_int_float_bool_types() -> None:
    def f(n: int, x: float, ok: bool) -> str:
        """Numbers."""
        return ""

    props = build_skill(f).parameters_schema["properties"]
    assert props["n"]["type"] == "integer"
    assert props["x"]["type"] == "number"
    assert props["ok"]["type"] == "boolean"


def test_optional_param_has_default_and_not_required() -> None:
    def f(name: str, fullscreen: bool = False) -> str:
        """Open."""
        return name

    schema = build_skill(f).parameters_schema
    assert schema["required"] == ["name"]
    assert schema["properties"]["fullscreen"]["default"] is False
    assert schema["properties"]["fullscreen"]["type"] == "boolean"


def test_no_params_yields_empty_properties() -> None:
    def f() -> str:
        """Now."""
        return "now"

    schema = build_skill(f).parameters_schema
    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema["required"] == []


# --------------------------------------------------------------------------- #
# Generic / Literal / Union                                                   #
# --------------------------------------------------------------------------- #


def test_literal_becomes_string_enum() -> None:
    def f(action: Literal["play", "pause", "next"]) -> str:
        """Control."""
        return action

    prop = build_skill(f).parameters_schema["properties"]["action"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["play", "pause", "next"]


def test_literal_of_ints_becomes_integer_enum() -> None:
    def f(n: Literal[1, 2, 3]) -> str:
        """Pick."""
        return str(n)

    prop = build_skill(f).parameters_schema["properties"]["n"]
    assert prop["type"] == "integer"
    assert prop["enum"] == [1, 2, 3]


def test_list_of_strings() -> None:
    def f(items: list[str]) -> str:
        """List."""
        return ""

    prop = build_skill(f).parameters_schema["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_optional_collapses_to_underlying_type() -> None:
    def f(name: str | None = None) -> str:
        """Optional."""
        return ""

    prop = build_skill(f).parameters_schema["properties"]["name"]
    # `T | None` should just become T's schema — the default None
    # carries the "this is optional" signal.
    assert prop["type"] == "string"
    assert prop["default"] is None
    assert build_skill(f).parameters_schema["required"] == []


def test_union_of_two_concrete_types_uses_anyof() -> None:
    def f(value: int | str) -> str:
        """Either."""
        return str(value)

    prop = build_skill(f).parameters_schema["properties"]["value"]
    assert "anyOf" in prop
    assert {"type": "integer"} in prop["anyOf"]
    assert {"type": "string"} in prop["anyOf"]


# --------------------------------------------------------------------------- #
# Error cases                                                                 #
# --------------------------------------------------------------------------- #


def test_missing_type_annotation_raises() -> None:
    def f(name) -> str:  # type: ignore[no-untyped-def]
        """X."""
        return ""

    with pytest.raises(ValueError, match="missing a type annotation"):
        build_skill(f)


def test_var_args_rejected() -> None:
    def f(*args: str) -> str:
        """X."""
        return ""

    with pytest.raises(ValueError, match="not supported"):
        build_skill(f)


def test_var_kwargs_rejected() -> None:
    def f(**kwargs: str) -> str:
        """X."""
        return ""

    with pytest.raises(ValueError, match="not supported"):
        build_skill(f)


def test_unsupported_type_raises() -> None:
    # ``bytes`` is a stdlib type the generator deliberately doesn't
    # handle — there's no clean JSON-schema mapping. Use it instead of
    # a local class so ``get_type_hints`` can resolve the annotation
    # under ``from __future__ import annotations``.
    def f(thing: bytes) -> str:
        """X."""
        return ""

    with pytest.raises(ValueError, match="unsupported parameter type"):
        build_skill(f)


# --------------------------------------------------------------------------- #
# Docstring handling                                                          #
# --------------------------------------------------------------------------- #


def test_docstring_summary_strips_parameters_block() -> None:
    def f(name: str) -> str:
        """Top summary line.

        Parameters
        ----------
        name:
            A name.
        """
        return name

    assert build_skill(f).description == "Top summary line."


def test_multi_line_summary_collapses_to_single_line() -> None:
    def f(name: str) -> str:
        """First line of summary
        that wraps onto a second line.

        Parameters
        ----------
        name:
            A name.
        """
        return name

    desc = build_skill(f).description
    assert "wraps" in desc and "summary" in desc
    assert "\n" not in desc


def test_one_line_docstring_with_no_params_section() -> None:
    def f(x: str) -> str:
        """Just a one-liner."""
        return x

    built = build_skill(f)
    assert built.description == "Just a one-liner."
    # Without a Parameters block the prop should still be schema-valid
    # — just without a per-param description.
    assert "description" not in built.parameters_schema["properties"]["x"]


def test_returns_section_does_not_pollute_param_block() -> None:
    def f(name: str) -> str:
        """Do.

        Parameters
        ----------
        name:
            A name.

        Returns
        -------
        A string.
        """
        return name

    props = build_skill(f).parameters_schema["properties"]
    assert set(props.keys()) == {"name"}
    assert props["name"]["description"] == "A name."


def test_param_description_handles_multi_line_indent() -> None:
    def f(name: str) -> str:
        """X.

        Parameters
        ----------
        name:
            First line.
            Second line that keeps the same parameter going.
        """
        return name

    desc = build_skill(f).parameters_schema["properties"]["name"]["description"]
    assert "First line." in desc
    assert "Second line" in desc


# --------------------------------------------------------------------------- #
# Tool schema shape                                                           #
# --------------------------------------------------------------------------- #


def test_tool_schema_shape_matches_openai_contract() -> None:
    def f(name: str) -> str:
        """Do."""
        return name

    schema = build_skill(f).tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "f"
    assert schema["function"]["description"] == "Do."
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "properties" in params and "required" in params


# --------------------------------------------------------------------------- #
# Decorator overrides                                                         #
# --------------------------------------------------------------------------- #


def test_skill_decorator_attaches_built_skill_to_function() -> None:
    reg = SkillRegistry()

    @skill(registry=reg)
    def hello(name: str) -> str:
        """Greet."""
        return name

    assert hello.skill.name == "hello"  # type: ignore[attr-defined]
    assert reg.get("hello") is not None


def test_skill_decorator_name_override() -> None:
    reg = SkillRegistry()

    @skill(registry=reg, name="renamed")
    def original(x: str) -> str:
        """X."""
        return x

    assert "renamed" in reg
    assert "original" not in reg


def test_skill_decorator_destructive_flag_propagates() -> None:
    reg = SkillRegistry()

    @skill(registry=reg, destructive=True)
    def boom(text: str) -> str:
        """X."""
        return text

    sk = reg.get("boom")
    assert sk is not None
    assert sk.destructive is True


def test_bare_skill_decorator_routes_to_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The no-paren ``@skill`` form goes through ``skill(fn)`` and
    # registers in ``default_registry``. Swap that out for a fresh
    # registry so the test doesn't leak state into / out of the
    # process-wide default.
    fresh = SkillRegistry()
    monkeypatch.setattr("friday.skills.registry.default_registry", fresh)

    @skill
    def f(name: str) -> str:
        """X."""
        return name

    assert f.skill.name == "f"  # type: ignore[attr-defined]
    assert "f" in fresh
