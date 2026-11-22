"""T2: PushToRegion and PickFromClutter, their tasks, and their geometry helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import ObjectSpec, Pose, Skill, TaskSpec
from skillforge.skills import PickFromClutter, PushToRegion
from skillforge.skills.grasp import DOWN, quat_from_axis_angle, quat_mul, quat_to_mat
from skillforge.skills.pick_clutter import HAND_BOTTOM, hand_clearance

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def can(name, x, y, h=0.05, r=0.025):
    return ObjectSpec(name=name, shape="cylinder", size=(r, h, 0.0), pose=Pose(pos=(x, y, h))), np.array([x, y, h])


def test_skills_implement_the_protocol():
    assert isinstance(PushToRegion(), Skill) and isinstance(PickFromClutter(), Skill)


def test_new_tasks_validate():
    push, clutter = load("franka_push_box_to_target"), load("franka_pick_cube_from_clutter")
    box = push.scene.object("box")
    assert 2 * min(box.size[:2]) > push.scene.robot.max_gripper_width  # too wide to grasp: must push
    assert len(clutter.scene.objects) == 4 and clutter.success.object == "cube"
    json.loads((TASKS / "franka_push_box_to_target.json").read_text())


def test_hand_clearance_sees_tall_neighbours_along_the_closing_axis_only():
    grasp = np.array([0.5, 0.0, 0.02])
    q_close_y = DOWN  # closing along world y
    q_close_x = quat_mul(quat_from_axis_angle([0, 0, 1], np.pi / 2), DOWN)
    assert abs(quat_to_mat(q_close_x)[0, 1]) > 0.99
    tall = [can("c", 0.5, 0.085)]  # 8.5 cm along +y
    assert hand_clearance(grasp, q_close_y, tall) < 0  # hand (0.104 half-length) overlaps it
    assert hand_clearance(grasp, q_close_x, tall) > 0  # across the hand (0.032 half-width): clear
    short = [can("s", 0.5, 0.085, h=0.01)]  # top at 2 cm, under the hand's underside
    assert grasp[2] + HAND_BOTTOM > 0.02
    assert hand_clearance(grasp, q_close_y, short) == np.inf


def test_push_quat_puts_the_closing_axis_across_the_push():
    sk = PushToRegion()
    for ang in (0.0, 0.7, np.pi / 2, -2.0):
        u = np.array([np.cos(ang), np.sin(ang)])
        m = quat_to_mat(sk._push_quat(u, DOWN))
        assert abs(float(m[:2, 1] @ u)) < 1e-6  # closing axis perpendicular to the push
        assert m[2, 2] < -0.99  # pointing down


def test_push_extent_follows_box_yaw():
    sk = PushToRegion()
    sk.reset(load("franka_push_box_to_target"))
    u = np.array([1.0, 0.0])
    assert sk._extent(u, np.array([1.0, 0, 0, 0])) == pytest.approx(0.05)
    q45 = quat_from_axis_angle([0, 0, 1], np.pi / 4)
    assert sk._extent(u, q45) == pytest.approx(0.05 * np.sqrt(2))


# -- MuJoCo -------------------------------------------------------------------


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.eval import evaluate
    from skillforge.sim import make_env

    return evaluate, make_env


def test_push_solves_seeds(mj):
    evaluate, make_env = mj
    task = load("franka_push_box_to_target")
    rep, eps = evaluate(make_env, PushToRegion(), task, [0, 1, 2])
    assert all(e.success for e in eps), [(e.seed, e.failure) for e in eps]


def test_clutter_skill_avoids_the_collisions_pick_place_makes(mj):
    evaluate, make_env = mj
    from skillforge.skills import PickPlace

    task = load("franka_pick_cube_from_clutter")
    # Seeds where PickPlace drives the hand into a can (seeds 3, 8, 11 locally). Exact outcomes can
    # differ across CPUs/MuJoCo builds, so require at least two collisions, not a fixed set.
    seeds = [3, 8, 11]
    _, base = evaluate(make_env, PickPlace(), task, seeds)
    assert sum(e.failure == "collision" for e in base) >= 2, [(e.seed, e.failure) for e in base]
    _, ours = evaluate(make_env, PickFromClutter(), task, seeds)
    assert all(e.success for e in ours), [(e.seed, e.failure) for e in ours]


def test_push_replans_if_the_object_is_seen_outside_after_retreating():
    from skillforge.contracts import ObjectState, Observation

    task = load("franka_push_box_to_target")
    sk = PushToRegion()
    sk.reset(task)
    outside = np.array([0.40, -0.10, 0.025])
    obs = Observation(t=0.0, qpos=np.zeros(7), qvel=np.zeros(7), ee_pos=np.array([0.5, 0.0, 0.3]),
                      ee_quat=DOWN.copy(), gripper_width=0.04,
                      objects={"box": ObjectState(pos=outside, quat=np.array([1.0, 0, 0, 0]))})  # fmt: skip
    sk.state, sk.park, sk.u = "done", obs.ee_pos.copy(), np.array([0.0, 1.0])
    for _ in range(sk.settle_steps + 2):
        sk.act(obs)
    assert sk.state in ("above", "down") and sk.replans == 1
