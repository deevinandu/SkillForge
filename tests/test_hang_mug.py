"""Mug hanging: the `mug` shape, the `object_hung` success kind, the closed pad gap, and HangMug."""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from skillforge.contracts import MugHandle, ObjectHung, ObjectSpec, Skill, SkillSource, TaskSpec
from skillforge.sim.mug import handle_opening, handle_threaded, mug_parts, peg_segment
from skillforge.skills import HangMug, load_skill
from skillforge.skills import hang_mug as hang_mug_module

TASKS = Path(__file__).resolve().parents[1] / "tasks"
MUG_TASKS = ("franka_hang_mug", "franka_hang_mug_handle_fixed", "franka_hang_mug_tree")


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def mug(**kw) -> ObjectSpec:
    return ObjectSpec(name="mug", shape="mug", size=(0.0325, 0.045, 0.003), **kw)


def peg_through(m: ObjectSpec, x: float, z: float, length: float = 0.07) -> ObjectSpec:
    """A cylinder along world y (= the mug's hole axis at yaw 0) through mug-frame point (x, 0, z)."""
    q = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)  # z -> +y
    return ObjectSpec(name="peg", shape="cylinder", size=(0.005, length / 2, 0.0), fixed=True,
                      pose={"pos": (x, 0.0, z), "quat": q})


# ---------------------------------------------------------------- contracts


def test_mug_spec_validation():
    assert mug().mug_handle() == MugHandle()
    with pytest.raises(ValidationError):
        ObjectSpec(name="b", shape="box", size=(0.02, 0.02, 0.02), handle=MugHandle())
    with pytest.raises(ValidationError):  # wall thicker than the radius
        ObjectSpec(name="m", shape="mug", size=(0.03, 0.045, 0.04))
    with pytest.raises(ValidationError):  # handle above the rim
        mug(handle=MugHandle(top=0.05))
    with pytest.raises(ValidationError):  # no opening
        MugHandle(reach=0.006, radius=0.004)


def test_object_hung_needs_a_fixed_cylinder_and_a_mug():
    t = load("franka_hang_mug").model_dump()
    t["success"]["support"] = "stand_base"  # a box
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(t)
    with pytest.raises(ValidationError):
        ObjectHung(object="mug", support="mug")


@pytest.mark.parametrize("name", MUG_TASKS)
def test_tasks_load(name):
    t = load(name)
    assert isinstance(t.success, ObjectHung)
    assert t.scene.object(t.success.support).fixed
    # the clear opening fits the 10 mm peg with room to spare
    x0, x1, z0, z1 = handle_opening(t.scene.object("mug"))
    assert x1 - x0 >= 0.02 and z1 - z0 >= 0.02


# ---------------------------------------------------------------- geometry


def test_mug_parts_mass_and_layout():
    m = mug(mass=0.2)
    parts = mug_parts(m)
    assert sum(p.mass for p in parts) == pytest.approx(0.2)
    caps = [p for p in parts if p.kind == "capsule"]
    assert caps and all(max(p.fromto[0], p.fromto[3]) > m.size[0] for p in caps)  # handle on +x
    walls = [p for p in parts if p.kind == "box"]
    # walls stop above the bottom face, so only the bottom disc touches a table
    assert all(p.pos[2] - p.size[2] > -m.size[1] for p in walls)


def test_handle_threaded():
    m = mug()
    x0, x1, z0, z1 = handle_opening(m)
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    ident = (1.0, 0.0, 0.0, 0.0)
    assert handle_threaded(m, (0, 0, 0), ident, peg_through(m, xc, zc))
    assert not handle_threaded(m, (0, 0, 0), ident, peg_through(m, 0.0, zc))  # through the cup, not the handle
    assert not handle_threaded(m, (0, 0, 0), ident, peg_through(m, xc, z1 + 0.01))  # above the loop
    # the peg must actually reach the loop's plane, not just point at it
    p = peg_through(m, xc, zc)
    p = p.model_copy(update={"pose": p.pose.model_copy(update={"pos": (xc, 0.06, zc)})})
    assert not handle_threaded(m, (0, 0, 0), ident, p)
    # a mug turned 90 degrees has its hole along world x: a peg along y no longer threads it
    q90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    assert not handle_threaded(m, (0, 0, 0), q90, peg_through(m, xc, zc))


def test_peg_segment_points_up_and_out():
    t = load("franka_hang_mug")
    root, tip = peg_segment(t.scene.object("peg"))
    post = t.scene.object("stand_post").pose.pos
    assert tip[2] > root[2]
    assert np.linalg.norm(tip[:2] - post[:2]) > np.linalg.norm(root[:2] - post[:2])


# ---------------------------------------------------------------- skill (no physics)


def test_skill_protocol_and_source():
    assert isinstance(HangMug(), Skill)
    src = SkillSource(name="hang_mug", version=HangMug.version, code=inspect.getsource(hang_mug_module),
                      entrypoint="HangMug")
    assert isinstance(load_skill(src), Skill)


def test_wrist_choice_keeps_the_turn_small():
    s = HangMug()
    s.reset(load("franka_hang_mug"))
    for psi in np.linspace(-math.pi, math.pi, 25):
        b0, psi_h = s._choose_yaw(float(psi))
        assert abs(b0 - s.wrist_home_yaw) <= math.pi / 2 + 1e-9
        assert abs(b0 + ((psi_h - psi + math.pi) % (2 * math.pi) - math.pi) - s.wrist_home_yaw) <= math.pi + 1e-9
        # hanging yaw puts the mug's y axis along the peg's horizontal direction
        assert abs(math.cos(psi_h - math.atan2(s.axis[1], s.axis[0]))) < 1e-9


# ---------------------------------------------------------------- physics


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    import mujoco

    from skillforge.sim.mujoco_env import MujocoEnv

    return mujoco, MujocoEnv


def _place(env, mujoco, pos, quat):
    a, v = env._obj_free["mug"], env._obj_dof["mug"]
    env.data.qpos[a : a + 3] = pos
    env.data.qpos[a + 3 : a + 7] = quat
    env.data.qvel[v : v + 6] = 0
    mujoco.mj_forward(env.model, env.data)


def _still(env, obs):
    from skillforge.contracts import Action

    return Action(mode="ee", ee_pos=obs.ee_pos.copy(), ee_quat=obs.ee_quat.copy(), gripper=0.0)


def _threaded_pose(task):
    from skillforge.sim.mug import quat_mat

    m = task.scene.object("mug")
    root, tip = peg_segment(task.scene.object(task.success.support))
    d = tip - root
    yaw = math.atan2(d[1], d[0]) - math.pi / 2
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    x0, x1, _, z1 = handle_opening(m)
    pos = root + 0.45 * d - quat_mat(q) @ np.array([(x0 + x1) / 2, 0, z1]) + [0, 0, 0.006]
    return pos, q


def test_hanging_counts_only_after_the_hold(mj):
    mujoco, MujocoEnv = mj
    task = load("franka_hang_mug")
    env = MujocoEnv(task)
    obs = env.reset(0)
    assert not any(env.step(_still(env, obs))[1].success for _ in range(10))  # on the table
    _place(env, mujoco, *_threaded_pose(task))
    flags = [env.step(_still(env, obs))[1].success for _ in range(80)]
    first = flags.index(True)
    assert first >= round(task.success.hold_s * task.control_hz) - 1
    assert all(flags[first:])
    touching = env._contact_state().touching["mug"]
    assert "peg" in touching and touching <= {o.name for o in task.scene.objects if o.fixed}


def test_mug_impaled_on_the_peg_is_not_hung(mj):
    """Cup dropped over the peg (peg inside the cup, not through the handle) rests on the stand
    but is not hanging by its handle."""
    mujoco, MujocoEnv = mj
    task = load("franka_hang_mug")
    env = MujocoEnv(task)
    obs = env.reset(0)
    root, tip = peg_segment(task.scene.object("peg"))
    m = task.scene.object("mug")
    pos = tip + np.array([0, 0, m.size[1] - 0.02])  # cup rim around the tip
    q = np.array([0.0, 1.0, 0.0, 0.0])  # upside down
    _place(env, mujoco, pos, q)
    assert not any(env.step(_still(env, obs))[1].success for _ in range(60))


@pytest.mark.parametrize("gap", [0.0, 0.003])
def test_closed_pad_gap(mj, gap):
    mujoco, MujocoEnv = mj
    from skillforge.contracts import Action

    env = MujocoEnv(load("franka_lift_cube"), closed_pad_gap=gap)
    obs = env.reset(0)
    for _ in range(30):
        obs, _ = env.step(Action(mode="ee", ee_pos=obs.ee_pos + [0, 0, 0.1], gripper=1.0))
    pads = [g for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] in env._finger_bodies and env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and env.model.geom_contype[g]]
    assert len(pads) == 2
    assert mujoco.mj_geomDistance(env.model, env.data, pads[0], pads[1], 0.05, None) == pytest.approx(gap, abs=3e-4)


def test_hang_mug_hangs_the_mug(mj):
    from skillforge.eval import evaluate
    from skillforge.sim import make_env

    task = load("franka_hang_mug")
    _, eps = evaluate(make_env, HangMug(), task, [0, 2])
    assert all(e.success for e in eps), [(e.seed, e.failure, e.failure_detail) for e in eps]
