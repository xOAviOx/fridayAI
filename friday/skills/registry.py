"""``@skill`` decorator + ``SkillRegistry``.

A *skill* is a typed Python function the agent loop is allowed to call.
``@skill`` introspects the function's signature, type hints, and
docstring at decoration time to produce the JSON Schema the LLM
provider needs in its tool-calling API — there are no hand-written
schemas anywhere in this codebase, and there will not be.

Schema generation is the contract this chunk has to honor; tests for
it live in ``tests/test_skill_schema.py`` (the spec calls those out
by name).

Design notes worth knowing
--------------------------
* The decorator is usable bare (``@skill``) or with overrides
  (``@skill(name="open_app", destructive=True)``).
* Bare ``@skill`` registers in ``default_registry``. Tests pass an
  explicit ``registry=`` to keep state hermetic.
* Schema is cached on the wrapped function as ``func.skill`` so the
  router can introspect without re-deriving.
* Type → JSON Schema mapping covers the primitives we actually use
  (str/int/float/bool, ``Literal[...]`` enums, ``list[T]``,
  ``T | None``). Anything weirder raises at decoration time — better
  than mailing a malformed tool-schema to the LLM at runtime.
* Docstring parsing supports the NumPy-style ``Parameters`` block the
  rest of this codebase already writes. If the docstring is just a
  one-liner, params get no description and that's fine.
"""

from __future__ import annotations

import inspect
import logging
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_args, get_origin

log = logging.getLogger(__name__)

# Recognised NumPy-style section headers. The parameter parser stops
# when it hits any of these so it doesn't accidentally treat a
# ``Returns`` block as another parameter.
_NUMPY_SECTION_HEADERS = frozenset(
    {
        "Parameters",
        "Returns",
        "Raises",
        "Yields",
        "Notes",
        "Examples",
        "References",
        "See Also",
    }
)


# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Skill:
    """One registered skill.

    Attributes
    ----------
    name:
        Tool name as exposed to the LLM. Defaults to the wrapped
        function's ``__name__`` and is overridable on the decorator.
    description:
        First paragraph of the docstring, with line wraps collapsed.
    parameters_schema:
        JSON Schema for the function's keyword arguments — the LLM
        provider sees this verbatim as the tool's ``parameters`` field.
    func:
        The wrapped callable. Always invoked with keyword args by the
        executor (matches how OpenAI / Groq deliver tool arguments).
    destructive:
        Hint consumed by the safety layer (chunk 4). Destructive skills
        require explicit confirmation; non-destructive ones can run in
        dry-run mode.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    func: Callable[..., Any]
    destructive: bool = False

    def tool_schema(self) -> dict[str, Any]:
        """Return the OpenAI / Groq tool-calling schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def call(self, **kwargs: Any) -> Any:
        """Invoke the underlying function with keyword arguments."""
        return self.func(**kwargs)


class SkillRegistry:
    """In-memory registry of skills.

    One instance is enough per agent loop. ``default_registry`` below
    collects every function decorated with bare ``@skill`` so import-
    time discovery works (``friday.skills.builtin`` is imported on
    package init, which registers the five starter skills). Tests use
    a fresh ``SkillRegistry()`` to stay isolated from default state.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Add a skill. Raises ``ValueError`` on name collision."""
        if skill.name in self._skills:
            raise ValueError(f"skill {skill.name!r} already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        """All registered skill names, sorted (handy for tests + HUD)."""
        return sorted(self._skills)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """All registered skills as OpenAI / Groq tool schemas."""
        return [s.tool_schema() for s in self._skills.values()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Look up ``name`` and call it with ``**arguments``.

        Raises ``KeyError`` for unknown skills. Unknown keyword arguments
        (hallucinated by smaller/fallback LLMs) are silently dropped with a
        warning so a single spurious key doesn't crash an otherwise valid call.
        """
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"unknown skill: {name!r}")

        # Filter to only the params the function actually accepts.
        known = set(inspect.signature(skill.func).parameters)
        filtered = {k: v for k, v in arguments.items() if k in known}
        dropped = set(arguments) - known
        if dropped:
            log.warning(
                "dispatch(%s): dropped unknown arg(s) %s — LLM hallucinated them",
                name,
                sorted(dropped),
            )
        return skill.call(**filtered)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    def __len__(self) -> int:
        return len(self._skills)


# Populated by ``@skill`` when no explicit registry is passed.
default_registry = SkillRegistry()


# --------------------------------------------------------------------------- #
# Decorator                                                                   #
# --------------------------------------------------------------------------- #


@typing.overload
def skill(func: Callable[..., Any], /) -> Callable[..., Any]: ...


@typing.overload
def skill(
    *,
    name: str | None = ...,
    description: str | None = ...,
    destructive: bool = ...,
    registry: SkillRegistry | None = ...,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def skill(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
    registry: SkillRegistry | None = None,
) -> Any:
    """Register a function as an agent-callable skill.

    Use as ``@skill`` for defaults, or ``@skill(name=..., destructive=...)``
    when overrides are needed. The derived :class:`Skill` is cached on
    the returned function as ``func.skill`` so callers can introspect
    without rebuilding.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        built = build_skill(
            fn,
            name=name,
            description=description,
            destructive=destructive,
        )
        target = registry if registry is not None else default_registry
        target.register(built)
        # Stash the Skill on the function so tests / introspection
        # don't have to go through a registry.
        fn.skill = built  # type: ignore[attr-defined]
        return fn

    return wrap(func) if func is not None else wrap


# --------------------------------------------------------------------------- #
# Schema generation                                                           #
# --------------------------------------------------------------------------- #


def build_skill(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
) -> Skill:
    """Build a :class:`Skill` from a Python function via introspection.

    Useful directly in tests; the ``@skill`` decorator is just a thin
    wrapper that also registers the result.
    """
    skill_name = name or fn.__name__
    raw_doc = inspect.getdoc(fn) or ""
    summary, param_docs = _parse_docstring(raw_doc)
    final_description = description or summary or skill_name

    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise ValueError(
            f"skill {skill_name!r}: failed to resolve type hints: {exc}"
        ) from exc

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"skill {skill_name!r}: *args / **kwargs not supported in tool "
                "schemas — list the parameters explicitly"
            )
        if pname not in hints:
            raise ValueError(
                f"skill {skill_name!r}: parameter {pname!r} is missing a type "
                "annotation (required to derive its JSON schema)"
            )

        prop_schema = _type_to_json_schema(hints[pname])
        if pname in param_docs:
            prop_schema["description"] = param_docs[pname]
        if param.default is not inspect.Parameter.empty:
            prop_schema["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop_schema

    parameters_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        # `additionalProperties: false` is the LLM's cue not to invent
        # extra args. Worth the four extra bytes per turn.
        "additionalProperties": False,
    }

    return Skill(
        name=skill_name,
        description=final_description,
        parameters_schema=parameters_schema,
        func=fn,
        destructive=destructive,
    )


# --------------------------------------------------------------------------- #
# Type → JSON Schema mapping                                                  #
# --------------------------------------------------------------------------- #


def _type_to_json_schema(t: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(t)
    args = get_args(t)

    if _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            raise ValueError(f"unsupported Union containing only None: {t!r}")
        if len(non_none) == 1:
            # ``T | None`` collapses to T's schema; the default value
            # (almost always None) handles "this is optional" semantics.
            return _type_to_json_schema(non_none[0])
        return {"anyOf": [_type_to_json_schema(a) for a in non_none]}

    if origin is Literal:
        enum = list(args)
        if not enum:
            raise ValueError("Literal[] with no values isn't a meaningful schema")
        # All members of a Literal should share a JSON type. Take the
        # first value's type as the schema's base.
        base = _python_value_type(enum[0])
        return {**base, "enum": enum}

    if origin in (list, tuple, set, frozenset):
        item_schema = _type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    return _python_type_to_schema(t)


def _python_type_to_schema(t: Any) -> dict[str, Any]:
    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is type(None):
        return {"type": "null"}
    raise ValueError(f"unsupported parameter type for JSON schema: {t!r}")


def _python_value_type(v: Any) -> dict[str, Any]:
    """Best-guess JSON type for a single Literal value."""
    # ``bool`` is a subclass of ``int`` in Python, so check it first.
    if isinstance(v, bool):
        return {"type": "boolean"}
    if isinstance(v, int):
        return {"type": "integer"}
    if isinstance(v, float):
        return {"type": "number"}
    if isinstance(v, str):
        return {"type": "string"}
    # Heterogeneous Literal — let the enum carry the value, drop the type.
    return {}


def _is_union(origin: Any) -> bool:
    """True for both ``typing.Union[...]`` and PEP-604 ``int | None``."""
    if origin is Union:
        return True
    union_type = getattr(types, "UnionType", None)
    return union_type is not None and origin is union_type


# --------------------------------------------------------------------------- #
# Docstring parsing                                                           #
# --------------------------------------------------------------------------- #

# NumPy-style header: a flush-left "Parameters" line directly followed
# by a row of dashes. We pin the dashes to >= 3 to avoid false positives
# on `-` bullet lists.
_NUMPY_HEADER_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z ]+)\s*\n-{3,}\s*$",
    re.MULTILINE,
)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {param_name: description})``.

    Summary is the prose before any structured section, with line wraps
    collapsed to single spaces so the LLM gets one clean sentence. Per-
    parameter descriptions come from a NumPy-style ``Parameters`` block;
    if there isn't one, the dict is empty and that's fine.
    """
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)

    headers = list(_NUMPY_HEADER_RE.finditer(doc))
    parameters_header = next(
        (m for m in headers if m.group("title") == "Parameters"),
        None,
    )

    if parameters_header is None:
        summary = doc
        param_block = ""
    else:
        summary = doc[: parameters_header.start()].rstrip()
        # Body of the parameters block: everything between the
        # underline and the next recognised section header.
        body_start = parameters_header.end()
        next_header = next(
            (
                m
                for m in headers
                if m.start() > parameters_header.end()
                and m.group("title") in _NUMPY_SECTION_HEADERS
            ),
            None,
        )
        body_end = next_header.start() if next_header else len(doc)
        param_block = doc[body_start:body_end].strip("\n")

    # Multi-line summaries collapse to a single line — the LLM doesn't
    # benefit from the manual line wrap and our token budget definitely
    # doesn't.
    summary = " ".join(s.strip() for s in summary.splitlines() if s.strip())

    param_docs = _parse_param_block(param_block) if param_block else {}
    return summary, param_docs


def _parse_param_block(block: str) -> dict[str, str]:
    """Parse the body of a NumPy-style ``Parameters`` section.

    Recognised line shapes (flush-left, no indent):

    * ``name:``               (this codebase's convention)
    * ``name : type``         (canonical NumPy)
    * ``name``                (NumPy with no type)

    Description lines are indented (any non-zero indent counts) and
    flow until the next flush-left name.
    """
    out: dict[str, str] = {}
    # Match a flush-left parameter heading. Allow trailing colon and an
    # optional `: type` suffix. We require a single bare identifier
    # followed by either a colon or end-of-line so we don't mis-parse a
    # short prose line as a param name.
    head_re = re.compile(
        r"^(?P<name>[A-Za-z_]\w*)\s*"
        r"(?:\s*:\s*[^\n]*?)?\s*:?\s*$"
    )

    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        text = " ".join(line.strip() for line in buf if line.strip()).strip()
        if text:
            out[current] = text
        current = None
        buf.clear()

    for line in block.splitlines():
        if not line.strip():
            buf.append("")
            continue
        is_indented = line[0] in (" ", "\t")
        if is_indented:
            buf.append(line)
            continue
        m = head_re.match(line.rstrip())
        if m:
            flush()
            current = m.group("name")
        else:
            # Flush-left line that doesn't look like a parameter heading
            # — assume the params block has ended.
            flush()
            break

    flush()
    return out


__all__ = [
    "Skill",
    "SkillRegistry",
    "build_skill",
    "default_registry",
    "skill",
]
