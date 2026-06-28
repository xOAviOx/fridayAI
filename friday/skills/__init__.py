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

# Side-effect import — the @skill decorators in these modules populate
# ``default_registry``. Kept after the registry imports so the registry
# exists by the time the decorators run.
from friday.skills import briefing as briefing  # noqa: F401  Phase 4.1
from friday.skills import builtin as builtin    # noqa: F401
from friday.skills import memory as memory      # noqa: F401  Phase 3.4
from friday.skills import spotify as spotify    # noqa: F401
from friday.skills import timers as timers      # noqa: F401  Phase 3.5
from friday.skills import weather as weather    # noqa: F401  Phase 3.2
from friday.skills import web as web            # noqa: F401  Phase 3.3

__all__ = [
    "Skill",
    "SkillRegistry",
    "build_skill",
    "default_registry",
    "skill",
]
