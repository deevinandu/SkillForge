"""Shared helpers for the T4 eval tests. Module-level so worker processes can unpickle them."""

import inspect

from skillforge import skills
from skillforge.contracts import Range, SkillSource, TaskSpec
from skillforge.eval.robustness import with_randomization
from skillforge.sim import StubEnv


def pick_place_source(version: str = "0.1.0", **overrides) -> SkillSource:
    """PickPlace as a SkillSource; `overrides` patch class attributes (e.g. tol=0.03)."""
    code = inspect.getsource(skills.pick_place)
    code = code.replace('version = "0.1.0"', f'version = "{version}"')
    for k, v in overrides.items():
        code += f"\nPickPlace.{k} = {v!r}\n"
    return SkillSource(name="pick_place", version=version, code=code, entrypoint="PickPlace")


def noisy(task: TaskSpec, noise_hi: float = 0.015) -> TaskSpec:
    """The stub task with enough observation noise that PickPlace fails some seeds."""
    rnd = task.scene.randomization.model_copy(update={"obs_noise_std": Range(lo=0.0, hi=noise_hi)})
    return with_randomization(task, rnd)


def make_stub(task: TaskSpec) -> StubEnv:
    return StubEnv(task)


class RaisingRenderEnv(StubEnv):
    """An env whose renderer is broken, like MuJoCo without a GL context."""

    def render(self, camera=None):
        raise RuntimeError("no GL context")
