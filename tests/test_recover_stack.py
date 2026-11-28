"""T2: knock-over recovery for the cylinder pyramid (StackWithRecovery, UprightCylinder)."""

from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import (
    ObjectSpec,
    ObjectState,
    Observation,
    Pose,
    TaskSpec,
    success_conditions,
)
from skillforge.skills import StackWithRecovery, UprightCylinder
from skillforge.skills.recover_stack import footprint_gap, lying, pick_gaps, upright

TASKS = Path(__file__).resolve().parents[1] / "tasks"
KNOCKED = "franka_stack_cylinder_pyramid_knocked"
UP = np.array([1.0, 0.0, 0.0, 0.0])
LYING_X = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # axis along world x


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def obs_with(objects: dict[str, tuple[tuple[float, float, float], np.ndarray]]) -> Observation:
    return Observation(
        t=0.0, qpos=np.zeros(7), qvel=np.zeros(7), ee_pos=np.array([0.5, 0.0, 0.3]), ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
        gripper_width=0.08, objects={n: ObjectState(np.asarray(p, dtype=float), q) for n, (p, q) in objects.items()},
    )


def test_orientation_helpers():
    assert upright(UP) and not lying(UP)
    assert lying(LYING_X) and not upright(LYING_X)


def test_footprint_gap_of_a_turned_box():
    wall = ObjectSpec(name="w", shape="box", size=(0.005, 0.2, 0.03), fixed=True,
                      pose=Pose(pos=(0.7, 0.0, 0.03), quat=(np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12))))
    assert footprint_gap(np.array([0.7, 0.0]), wall) == 0.0  # inside
    assert footprint_gap(np.array([0.6, 0.0]), wall) == pytest.approx(0.1 * np.cos(np.pi / 6) - 0.005, abs=1e-6)


def test_pick_gaps_see_tall_neighbours_under_the_hand_body_and_lying_ones_only_by_the_fingers():
    cyl = ObjectSpec(name="c", shape="cylinder", size=(0.0254, 0.0419, 0.0))
    g = np.array([0.5, 0.0, 0.0254])  # grasping a cylinder lying along x: fingers close along y
    closing, axis = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    # a standing cylinder 9 cm to the side: clear of the fingers (6.6 cm), not of the hand body (10.4 cm)
    gaps = pick_gaps(g, closing, axis, 0.0254, obs_with({"n": ((0.5, 0.09, 0.0419), UP)}), {"n": cyl})
    assert gaps["n"] < 0
    # lying there instead, it is lower than the hand body's underside: only the fingers matter
    gaps = pick_gaps(g, closing, axis, 0.0254, obs_with({"n": ((0.5, 0.12, 0.0254), LYING_X)}), {"n": cyl})
    assert gaps["n"] > 0
    gaps = pick_gaps(g, closing, axis, 0.0254, obs_with({"n": ((0.5, 0.25, 0.0419), UP)}), {"n": cyl})
    assert gaps["n"] > 0.1


def test_knocked_task_border_encloses_the_work():
    t = load(KNOCKED)
    border = [o for o in t.scene.objects if o.fixed]
    assert len(border) == 12  # a ramp and a wall per side
    assert t.scene.randomization.knockdown is not None
    r = t.scene.object("cyl_1").size[0]
    spots = [np.asarray(t.scene.region(c.region).center[:2]) for c in success_conditions(t.success)]
    starts = [np.asarray(o.pose.pos[:2]) for o in t.scene.objects if not o.fixed]
    for xy in spots + starts:
        assert min(footprint_gap(xy, b) for b in border) > r + 0.03  # with room for the fingers


def _planner():
    t = load(KNOCKED)
    sk = StackWithRecovery()
    sk.reset(t)
    return t, sk


def _spot(t, c, dz=0.0):
    return tuple(np.asarray(t.scene.region(c.region).center) + [0.0, 0.0, dz])


def test_planner_stands_up_a_lying_cylinder_instead_of_placing_it():
    t, sk = _planner()
    conds = sk.conds
    objs = {c.object: (_spot(t, c), UP) for c in conds[:3]}  # the bottom row is built
    objs.update({c.object: ((0.42, -0.2 + 0.1 * i, 0.0254), LYING_X) for i, c in enumerate(conds[3:])})
    sk.reset(t)
    sk.act(obs_with(objs))  # records positions
    nxt, _ = sk._next(obs_with(objs))
    assert isinstance(nxt, UprightCylinder) and nxt.obj == conds[3].object


def test_planner_keeps_building_when_one_object_is_out_of_attempts():
    t, sk = _planner()
    c1, c2 = sk.conds[0], sk.conds[1]
    objs = {c1.object: ((0.42, -0.25, 0.0254), LYING_X)}  # lying and out of attempts
    objs.update({c.object: ((0.40 + 0.06 * (i % 3), -0.1 + 0.07 * (i // 3), 0.0419), UP) for i, c in enumerate(sk.conds[1:])})
    sk.attempts[c1.object] = sk.max_attempts
    sk.act(obs_with(objs))
    nxt, sub = sk._next(obs_with(objs))
    assert sub.success == c2  # the next bottom spot, not a stop and not the spots on top of c1


def test_planner_re_places_a_free_cylinder_just_outside_its_spot_but_not_one_holding_others():
    t, sk = _planner()
    bottom, second = sk.conds[:3], sk.conds[3:5]
    half = t.scene.region(bottom[0].region).half_extents[0]
    objs = {c.object: (_spot(t, c), UP) for c in bottom}
    objs[bottom[0].object] = (tuple(np.asarray(_spot(t, bottom[0])) + [half - 0.001, 0, 0]), UP)  # inside, by 1 mm
    for c in sk.conds[3:]:
        objs[c.object] = ((0.40, -0.25 + 0.06 * sk.conds.index(c), 0.0419), UP)
    for _ in range(sk.avg_window):
        sk.act(obs_with(objs))
    nxt, sub = sk._next(obs_with(objs))
    assert sub.success == bottom[0]  # not 1.5 mm inside yet: re-place it before building on it
    # once the second row stands on it, it is left alone
    objs.update({c.object: (_spot(t, c), UP) for c in second})
    sk.seen.clear()
    sk.act(obs_with(objs))
    nxt, sub = sk._next(obs_with(objs))
    assert sub.success == sk.conds[5]


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.eval import evaluate
    from skillforge.sim import make_env

    return evaluate, make_env


@pytest.mark.parametrize("seed", [0, 9])
def test_rebuilds_the_knocked_pyramid(mj, seed):
    # two cylinders knocked over each (seed 0 at step 784, with the hand touching a cylinder:
    # the env must refresh its contacts after the knock-over; seed 9 at step 1313)
    evaluate, make_env = mj
    t = load(KNOCKED)
    _, (ep,) = evaluate(make_env, StackWithRecovery(), t, [seed])
    assert ep.failure != "exception", ep.failure_detail
    assert ep.success, (ep.failure, ep.failure_detail)
    assert sum(e.startswith("knockdown:") for e in ep.events) == 2
