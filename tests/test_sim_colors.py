"""T1: object colours, per-episode colour shuffling, task-requested cameras, color_in_region."""

import json

import numpy as np
import pytest

from skillforge.contracts import TaskSpec
from skillforge.sim import is_success


def _task(stub_task, shuffle=False, cameras=(), success=None) -> TaskSpec:
    d = json.loads(stub_task.model_dump_json())
    cube = d["scene"]["objects"][0]
    d["scene"]["objects"] = [
        {**cube, "name": f"c{i}", "color": col, "pose": {"pos": [0.45 + 0.05 * i, -0.15, cube["pose"]["pos"][2]]}}
        for i, col in enumerate(("red", "red", "blue", "blue"))
    ]
    d["scene"]["randomization"]["shuffle_colors"] = shuffle
    d["scene"]["obs_cameras"] = list(cameras)
    d["success"] = success or {"kind": "color_in_region", "color": "red", "region": "bin"}
    return TaskSpec.model_validate(d)


def test_color_in_region_uses_the_episode_colours(stub_task):
    t = _task(stub_task)
    c = np.asarray(t.scene.region("bin").center, dtype=float)
    far = c + [0.5, 0, 0]
    pos = {"c0": c, "c1": c, "c2": far, "c3": far}
    assert is_success(t, pos, pos)  # spec colours: c0, c1 are red and in the bin
    assert not is_success(t, pos, pos, colors={"c0": "red", "c1": "blue", "c2": "red", "c3": "blue"})
    assert is_success(t, pos, pos, colors={"c0": "red", "c1": "red", "c2": "blue", "c3": "blue"})


@pytest.fixture(scope="module")
def mujoco_env():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.sim.mujoco_env import COLOR_RGBA, MujocoEnv

    return MujocoEnv, COLOR_RGBA


def _franka(stub_task, **kw) -> TaskSpec:
    from pathlib import Path

    base = TaskSpec.model_validate_json((Path(__file__).resolve().parents[1] / "tasks" / "franka_pick_cube_to_bin.json").read_text())
    t = _task(stub_task, **kw)
    return t.model_copy(update={"scene": t.scene.model_copy(update={"robot": base.scene.robot, "regions": base.scene.regions})})


def test_spec_colours_are_rendered_and_shuffles_are_seeded(stub_task, mujoco_env):
    MujocoEnv, COLOR_RGBA = mujoco_env
    env = MujocoEnv(_franka(stub_task, shuffle=False))
    try:
        env.reset(0)
        assert env._colors == {"c0": "red", "c1": "red", "c2": "blue", "c3": "blue"}
        np.testing.assert_allclose(env.model.geom_rgba[env.model.geom("c2").id], COLOR_RGBA["blue"])
    finally:
        env.close()
    env = MujocoEnv(_franka(stub_task, shuffle=True))
    try:
        draws = []
        for seed in range(8):
            env.reset(seed)
            draws.append(tuple(env._colors[f"c{i}"] for i in range(4)))
            assert sorted(draws[-1]) == ["blue", "blue", "red", "red"]  # a permutation
            for i in range(4):
                np.testing.assert_allclose(env.model.geom_rgba[env.model.geom(f"c{i}").id], COLOR_RGBA[draws[-1][i]])
            assert env.randomization_sample()["c0.color"] in (0.0, 1.0)  # red or blue's palette index
        env.reset(3)
        assert tuple(env._colors[f"c{i}"] for i in range(4)) == draws[3]  # same seed, same colours
        assert len(set(draws)) > 1
    finally:
        env.close()


def test_task_cameras_reach_the_observation(stub_task, mujoco_env):
    MujocoEnv, _ = mujoco_env
    env = MujocoEnv(_franka(stub_task, cameras=("wrist",)))
    try:
        assert env.obs_cameras == ("wrist",)
        obs = env.reset(0)
        if env.render("wrist") is None:
            pytest.skip("no GL context to render with")
        assert obs.images["wrist"].ndim == 3 and obs.images["wrist"].dtype == np.uint8
    finally:
        env.close()


def test_success_judged_on_the_drawn_colours(stub_task, mujoco_env):
    import mujoco

    from skillforge.contracts import Action

    MujocoEnv, _ = mujoco_env
    t = _franka(stub_task, shuffle=True)
    env = MujocoEnv(t)
    c = t.scene.region("bin").center
    try:
        obs = env.reset(1)
        reds = [n for n, col in env._colors.items() if col == "red"]
        hold = Action(ee_pos=obs.ee_pos, ee_quat=obs.ee_quat)
        for k, name in enumerate(reds):  # the episode's red objects into the bin, one at a time
            a = env.model.joint(f"{name}_free").qposadr[0]
            env.data.qpos[a : a + 3] = [c[0] + (0.03 if k else -0.03), c[1], 0.02]
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            _, info = env.step(hold)
            assert info.success == (k == len(reds) - 1)  # only once every red one is in
    finally:
        env.close()
