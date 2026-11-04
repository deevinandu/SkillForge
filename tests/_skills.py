"""Skill source snippets for agent tests. Each is a complete file, as an agent would write it."""

from skillforge.contracts import SkillSource

GOOD = '''
from skillforge.skills.pick_place_v01 import PickPlaceV01 as _Base

class Skill(_Base):
    name = "t"
    version = "x"
'''

# Gives up whenever the cube's first observed y is below its nominal start: about half the seeds.
PARTIAL = '''
import numpy as np
from skillforge.contracts import Action
from skillforge.skills.pick_place_v01 import PickPlaceV01 as _Base

class Skill(_Base):
    name = "t"
    version = "x"

    def reset(self, task):
        super().reset(task)
        self.give_up = None

    def act(self, obs):
        if self.give_up is None:
            self.give_up = bool(obs.objects[self.obj].pos[1] < -0.1)
        if self.give_up:
            return Action(ee_pos=obs.ee_pos, gripper=0.0)
        return super().act(obs)
'''

NOOP = '''
from skillforge.contracts import Action

class Skill:
    name = "t"
    version = "x"

    def reset(self, task):
        pass

    def act(self, obs):
        return Action(ee_pos=obs.ee_pos, gripper=0.0)
'''

BROKEN = "class Skill(:\n    pass\n"

HANG = NOOP.replace("return Action(ee_pos=obs.ee_pos, gripper=0.0)", "while True:\n            pass")

NETWORK = NOOP.replace(
    "    def reset(self, task):\n        pass",
    "    def __init__(self):\n        import socket\n        socket.create_connection(('example.com', 80), timeout=1)\n\n"
    "    def reset(self, task):\n        pass",
)

LEAK_CHECK = NOOP.replace(
    "    def reset(self, task):\n        pass",
    "    def reset(self, task):\n        import os\n"
    "        if os.environ.get('ANTHROPIC_API_KEY'):\n            raise RuntimeError('secret leaked into sandbox')",
)


def src(code: str, version: str = "0.1.0") -> SkillSource:
    return SkillSource(name="t", version=version, code=code, entrypoint="Skill", rationale="test")
