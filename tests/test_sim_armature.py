"""T1: arm servo bang-bang at the force limits (menagerie armature 0.1) is gone at the 0.5 floor."""

from pathlib import Path

import numpy as np
import pytest

from skillforge.sim import default_mujoco_gl

default_mujoco_gl()
mujoco = pytest.importorskip("mujoco")

from skillforge.cli import load_seed_skill  # noqa: E402
from skillforge.contracts import TaskSpec  # noqa: E402
from skillforge.eval.harness import run_episode  # noqa: E402
from skillforge.sim.mujoco_env import MujocoEnv  # noqa: E402
from skillforge.skills import load_skill  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BALL_SKILL = ROOT / "results/2026-11-05-claude-code-repeats/ball_r1/pick_place_0.2.1.py"


def bang_bang_substeps(env: MujocoEnv, skill, task: TaskSpec, seed: int, monkeypatch) -> int:
    """Substeps on which an arm actuator at its force limit flips to the opposite limit."""
    lim = env.model.actuator_forcerange[env._arm_act, 1]
    real = mujoco.mj_step
    count, prev = [0], [None]

    def step(m, d, nstep=1):
        for _ in range(nstep):
            real(m, d)
            f = d.actuator_force[env._arm_act]
            sat, sign = np.abs(f) >= lim - 1e-3, np.sign(f)
            if prev[0] is not None and np.any(sat & prev[0][0] & (sign != prev[0][1])):
                count[0] += 1
            prev[0] = (sat, sign)

    monkeypatch.setattr(mujoco, "mj_step", step)
    run_episode(env, skill, task, seed)
    monkeypatch.setattr(mujoco, "mj_step", real)
    return count[0]


def test_default_armature_floor():
    task = TaskSpec.model_validate_json((ROOT / "tasks/franka_pick_cube_to_bin.json").read_text())
    env = MujocoEnv(task)
    assert np.all(env.model.dof_armature[env._arm_v] >= 0.5)
    env.reset(3)  # randomization re-applies masses etc. but must keep the armature
    assert np.all(env.model.dof_armature[env._arm_v] >= 0.5)
    env.close()


@pytest.mark.skipif(not BALL_SKILL.exists(), reason="agent ball skill not in this checkout")
def test_no_servo_bang_bang_with_the_armature_floor(monkeypatch):
    task = TaskSpec.model_validate_json((ROOT / "tasks/franka_pick_ball_to_bin.json").read_text())
    skill = load_skill(load_seed_skill(str(BALL_SKILL)))
    old = MujocoEnv(task, arm_armature=0.1)
    new = MujocoEnv(task)
    try:
        assert bang_bang_substeps(old, skill, task, 8, monkeypatch) > 200  # the artifact, reproduced
        assert bang_bang_substeps(new, skill, task, 8, monkeypatch) < 20
    finally:
        old.close()
        new.close()
