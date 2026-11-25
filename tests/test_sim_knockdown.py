"""T1: the knockdown disturbance (objects laid on their side mid-episode, seeded)."""

import json
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Action, TaskSpec

TASKS = Path(__file__).resolve().parents[1] / "tasks"


@pytest.fixture(scope="module")
def MujocoEnv():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.sim.mujoco_env import MujocoEnv

    return MujocoEnv


def _task(knock: dict | None) -> TaskSpec:
    d = json.loads((TASKS / "franka_stack_block_pyramid.json").read_text())
    d["scene"]["randomization"]["knockdown"] = knock
    return TaskSpec.model_validate(d)


def _run(env, seed, n):
    obs = env.reset(seed)
    hold = Action(ee_pos=obs.ee_pos, ee_quat=obs.ee_quat)
    events = []
    for _ in range(n):
        _, info = env.step(hold)
        events += info.events
    return events


def _lying(env, name) -> bool:
    return abs(env.data.xmat[env.model.body(name).id].reshape(3, 3)[2, 2]) < 0.2


def test_knockdown_lays_objects_down_at_rest_and_is_seeded(MujocoEnv):
    t = _task({"at_step": {"lo": 3, "hi": 3}, "count": {"lo": 2, "hi": 2}, "radius": 0.08})
    env = MujocoEnv(t)
    try:
        start = {o.name: np.array(o.pose.pos) for o in t.scene.objects}
        events = _run(env, 4, 30)
        knocked = [e.split(":")[1] for e in events if e.startswith("knockdown:")]
        assert len(knocked) == 2 and all(_lying(env, n) for n in knocked)
        assert sum(_lying(env, o.name) for o in t.scene.objects) == 2  # nothing else was touched
        for n in knocked:
            p = env.data.xpos[env.model.body(n).id]
            assert np.linalg.norm(p[:2] - start[n][:2]) < 0.08 + 0.03  # near where it stood (+ jitter)
            assert np.linalg.norm(env.data.cvel[env.model.body(n).id]) < 0.05  # at rest
            assert env.randomization_sample()[f"{n}.knocked"] == 1.0
        s = env.randomization_sample()
        assert s["knockdown_step"] == 3 and s["knockdown_count"] == 2
        again = [e for e in _run(env, 4, 30) if e.startswith("knockdown:")]
        assert again == [f"knockdown:{n}" for n in knocked]  # same seed, same knock
    finally:
        env.close()


def test_no_knockdown_by_default(MujocoEnv):
    env = MujocoEnv(_task(None))
    try:
        events = _run(env, 4, 10)
        assert not any(e.startswith("knockdown:") for e in events)
        assert "knockdown_step" not in env.randomization_sample()
    finally:
        env.close()
