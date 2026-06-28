"""Skill registry + builtin skills.

Importing this package triggers registration of every builtin skill
in :data:`default_registry`, so the agent loop can do::

    from friday.skills import default_registry
    schemas = default_registry.tool_schemas()

without an extra discovery step. Tests that need a clean slate
construct their own :class:`SkillRegistry`.

The ``destructive=True`` flag on a skill is the marker the safety
layer (chunk 4) reads when deciding whether to require confirmation.
"""

from friday.skills.registry import (
    Skill,
    SkillRegistry,
    build_skill,
    default_registry,
    skill,
)

# Side-effect import — the @skill decorators in this module populate
# ``default_registry``. Kept after the registry imports so the registry
# exists by the time the decorators run.
from friday.skills import builtin as builtin  # noqa: F401

__all__ = [
    "Skill",
    "SkillRegistry",
    "build_skill",
    "default_registry",
    "skill",
]
