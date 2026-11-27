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


def test_knocking_a_support_brings_down_what_stands_on_it(MujocoEnv):
    import mujoco

    t = _task({"at_step": {"lo": 2, "hi": 2}, "count": {"lo": 1, "hi": 1}})
    env = MujocoEnv(t)
    try:
        obs = env.reset(0)
        # build a 2-high stack by hand: block_2 on block_1, everything else far away
        d = env.data
        for i, name in enumerate(o.name for o in t.scene.objects):
            a = env.model.joint(f"{name}_free").qposadr[0]
            d.qpos[a : a + 7] = [0.45, -0.3 + 0.1 * i, 0.02, 1, 0, 0, 0]
        a1, a2 = (env.model.joint(f"block_{k}_free").qposadr[0] for k in (1, 2))
        d.qpos[a1 : a1 + 3] = [0.5, 0.15, 0.02]
        d.qpos[a2 : a2 + 3] = [0.5, 0.15, 0.06]
        d.qvel[:] = 0
        mujoco.mj_forward(env.model, d)
        env._knock_rng = __import__("numpy").random.default_rng(0)
        knocked = [e.split(":")[1] for e in env._knock_over(1, 0.08, names=["block_1"])]
        assert knocked == ["block_1", "block_2"]  # the one standing on it came down too
        hold = Action(ee_pos=obs.ee_pos, ee_quat=obs.ee_quat)
        for _ in range(20):
            env.step(hold)
        for n in knocked:
            assert env.data.xpos[env.model.body(n).id][2] < 0.03  # on the table
    finally:
        env.close()


def test_knockdown_keeps_clear_of_fixed_objects(MujocoEnv):
    # Fixed objects (walls here) are obstacles, never candidates: knocked objects land clear of them.
    d = json.loads((TASKS / "franka_stack_block_pyramid.json").read_text())
    d["scene"]["randomization"]["knockdown"] = {"at_step": {"lo": 5, "hi": 5}, "count": {"lo": 2, "hi": 2}, "radius": 0.12}
    walls = [("wall_a", (0.50, -0.20)), ("wall_b", (0.62, -0.10)), ("wall_c", (0.40, -0.05))]
    for name, (x, y) in walls:
        d["scene"]["objects"].append({"name": name, "shape": "box", "size": [0.005, 0.05, 0.03], "fixed": True,
                                      "pose": {"pos": [x, y, 0.03]}})
    env = MujocoEnv(TaskSpec.model_validate(d))
    try:
        for seed in range(6):
            events = _run(env, seed, 15)
            knocked = [e.split(":", 1)[1] for e in events if e.startswith("knockdown:")]
            assert knocked and not any(n.startswith("wall") for n in knocked)
            for n in knocked:
                xy = env.data.xpos[env.model.body(n).id][:2]
                for _, (x, y) in walls:  # footprint of a lying block: its half diagonal, 2.9 cm
                    gap = np.hypot(max(abs(xy[0] - x) - 0.005, 0.0), max(abs(xy[1] - y) - 0.05, 0.0))
                    assert gap > 0.028, (seed, n, gap)
    finally:
        env.close()


def test_cylinders_have_rolling_friction(MujocoEnv):
    # condim 6 for cylinders: a lying cylinder that is set rolling slows down and stops.
    import mujoco

    d = json.loads((TASKS / "franka_pick_cylinder_to_bin.json").read_text())  # a 2 cm radius can
    env = MujocoEnv(TaskSpec.model_validate(d))
    try:
        obs = env.reset(0)
        m, dt = env.model, env.data
        assert m.geom_condim[m.geom("can").id] == 6
        a, v = m.joint("can_free").qposadr[0], m.joint("can_free").dofadr[0]
        dt.qpos[a : a + 3] = [0.45, 0.10, 0.0205]
        dt.qpos[a + 3 : a + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # axis along world y
        dt.qvel[v : v + 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dt.qvel[v] = 0.10  # rolling along +x: spin about the axis to match
        dt.qvel[v + 3 : v + 6] = np.array([0.0, 0.0, -0.10 / 0.02])  # body z (the axis) is world -y
        mujoco.mj_forward(m, dt)
        hold = Action(ee_pos=obs.ee_pos, ee_quat=obs.ee_quat)
        for _ in range(150):  # 7.5 s
            env.step(hold)
        assert dt.qpos[a] > 0.50  # it rolled (at condim 4 it would roll on to the table's edge) ...
        assert np.linalg.norm(dt.qvel[v : v + 3]) < 0.01  # ... and stopped
    finally:
        env.close()
