"""Load an agent-written SkillSource into a live Skill object.

Agent code runs in-process for now. T3 moves this into a subprocess sandbox before any
untrusted or networked code is executed.
"""

from __future__ import annotations

import types

from skillforge.contracts import Skill, SkillSource


def load_skill(src: SkillSource) -> Skill:
    mod = types.ModuleType(f"skillforge_dyn_{src.name}_{src.version.replace('.', '_')}")
    exec(compile(src.code, f"<skill {src.name}@{src.version}>", "exec"), mod.__dict__)
    cls = getattr(mod, src.entrypoint)
    skill = cls()
    if not isinstance(skill, Skill):
        raise TypeError(f"{src.entrypoint} does not implement the Skill protocol (needs name, version, reset, act)")
    return skill
