"""T2: the six-block pyramid task and PlaceInOrder (PickPlace per block, each to an exact spot)."""

from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Skill, TaskSpec, success_conditions
from skillforge.skills import PickPlace, PlaceInOrder
from skillforge.skills.place_in_order import SpotPlace

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def test_pyramid_geometry():
    t = load("franka_stack_block_pyramid")
    conds = success_conditions(t.success)
    spots = {c.object: np.asarray(t.scene.region(c.region).center) for c in conds}
    h = t.scene.object("block_1").size[2]
    rows = {z: sorted(n for n, p in spots.items() if abs(p[2] - z) < 1e-6) for z in (h, 3 * h, 5 * h)}
    assert [len(r) for r in rows.values()] == [3, 2, 1]  # 3 on the base, 2 on them, 1 on top
    assert [c.object for c in conds][:3] == rows[h]  # the bottom row is placed first
    y = lambda n: spots[n][1]
    gap = y("block_2") - y("block_1") - 2 * h
    assert gap > 0  # room between neighbours for placement error
    # each upper block straddles the gap between the two below it
    assert y("block_4") == pytest.approx((y("block_1") + y("block_2")) / 2)
    assert y("block_5") == pytest.approx((y("block_2") + y("block_3")) / 2)
    assert y("block_6") == pytest.approx((y("block_4") + y("block_5")) / 2)
    # a spot is too short to hold a block that is not at its stack height
    assert all(t.scene.region(c.region).half_extents[2] < h for c in conds)


def test_place_in_order_setup():
    t = load("franka_stack_block_pyramid")
    sk = PlaceInOrder()
    sk.reset(t)
    assert isinstance(sk, Skill) and len(sk.skills) == 6
    for s, c in zip(sk.skills, success_conditions(t.success), strict=True):
        assert isinstance(s, SpotPlace) and s.closing_axis == (1.0, 0.0, 0.0)  # across the row (along y)
        np.testing.assert_allclose(s.place_obj, t.scene.region(c.region).center)  # the spot, not "floor + half"
        assert s.tol < PickPlace.tol
    assert PickPlace.closing_axis is None  # the library default is unchanged
    with pytest.raises(ValueError):
        PlaceInOrder().reset(load("franka_pick_cube_to_bin"))


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.eval import evaluate
    from skillforge.sim import make_env

    return evaluate, make_env


def test_builds_the_pyramid_bottom_up(mj):
    evaluate, make_env = mj
    t = load("franka_stack_block_pyramid")
    _, (ep,) = evaluate(make_env, PlaceInOrder(), t, [0])
    assert ep.success, (ep.failure, ep.failure_detail)
    grasps = [e.split(":")[1] for e in ep.events if e.startswith("grasp:")]
    assert grasps == [f"block_{i}" for i in range(1, 7)]
