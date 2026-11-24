"""T2: the tube-into-rack and rack-fill tasks, TubeIntoRack (a tight insertion, about 2 mm clearance
per side) and FillRack (several tubes, one after another)."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Skill, SkillSource, TaskSpec
from skillforge.skills import TubeIntoRack, load_skill
from skillforge.skills import tube_rack as tube_rack_module

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def test_task_geometry():
    t = load("franka_tube_into_rack")
    tube, slot = t.scene.object("tube"), t.scene.region("slot_middle")
    rack = [o for o in t.scene.objects if o.fixed]
    assert len(rack) == 6 and not tube.fixed
    # the slot is the gap between the separators either side of it, and the tube fits with ~2 mm a side
    seps = sorted(o.pose.pos[1] for o in rack if o.name.startswith("rack_sep"))
    below = max(y for y in seps if y < slot.center[1])
    above = min(y for y in seps if y > slot.center[1])
    t_half = rack[-1].size[1]
    hole = (above - t_half) - (below + t_half)
    assert hole - 2 * tube.size[0] == pytest.approx(0.004, abs=1e-6)
    # seated means standing on the table inside the slot; lying across the rack top (centre at
    # rack top + radius) must not count
    assert slot.center[2] == pytest.approx(tube.size[1])
    rack_top = max(o.pose.pos[2] + o.size[2] for o in rack)
    assert rack_top + tube.size[0] > slot.center[2] + slot.half_extents[2]


def test_skill_protocol_and_source():
    assert isinstance(TubeIntoRack(), Skill)
    src = SkillSource(name="tube_into_rack", version="0.2.0", code=inspect.getsource(tube_rack_module), entrypoint="TubeIntoRack")
    assert load_skill(src).version == "0.2.0"


def test_rack_top_from_fixed_objects():
    sk = TubeIntoRack()
    sk.reset(load("franka_tube_into_rack"))
    assert sk.rack_top == pytest.approx(0.05) and sk.half_len == pytest.approx(0.05)


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.eval import evaluate
    from skillforge.sim import make_env

    return evaluate, make_env


def test_tube_goes_into_the_slot(mj):
    evaluate, make_env = mj
    task = load("franka_tube_into_rack")
    _, eps = evaluate(make_env, TubeIntoRack(), task, [0, 1, 2])
    assert all(e.success for e in eps), [(e.seed, e.failure) for e in eps]


def test_pick_place_mostly_fails_where_tube_into_rack_succeeds(mj):
    """PickPlace grips the tube 2 cm from its top and drops it on the carry (30/100 on seeds 0-99)."""
    evaluate, make_env = mj
    from skillforge.skills import PickPlace

    task = load("franka_tube_into_rack")
    seeds = [0, 1, 6, 7]
    _, base = evaluate(make_env, PickPlace(), task, seeds)
    assert sum(not e.success for e in base) >= 2, [(e.seed, e.failure) for e in base]


def test_lying_tube_is_reported_not_grasped(mj):
    evaluate, make_env = mj
    from skillforge.contracts import Pose

    task = load("franka_tube_into_rack")
    tube = task.scene.object("tube")
    lying = tube.model_copy(update={"pose": Pose(pos=(0.5, -0.15, tube.size[0]), quat=(np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0))})
    objs = tuple(lying if o.name == "tube" else o for o in task.scene.objects)
    t2 = task.model_copy(update={"scene": task.scene.model_copy(update={"objects": objs})})
    from skillforge.eval.harness import run_episode

    env, sk = make_env(t2), TubeIntoRack()
    try:
        ep = run_episode(env, sk, t2, 0)
    finally:
        env.close()
    assert not ep.success and any("not upright" in line for line in sk.ctx.log), sk.ctx.log


def test_fill_task_and_skill_setup():
    from skillforge.contracts import AllOf, success_conditions
    from skillforge.skills import FillRack

    t = load("franka_fill_tube_rack")
    assert isinstance(t.success, AllOf)
    conds = success_conditions(t.success)
    assert [c.object for c in conds] == ["tube_a", "tube_b", "tube_c"]
    sk = FillRack()
    sk.reset(t)
    assert isinstance(sk, Skill) and len(sk.skills) == 3
    # slots run along y, so every tube is gripped with the fingers closing along x, and released narrow
    assert all(s.closing_axis == (1.0, 0.0, 0.0) and s.release_margin == sk.release_margin for s in sk.skills)
    assert all(s._release_cmd() > 0.0 for s in sk.skills)
    with pytest.raises(ValueError):
        FillRack().reset(load("franka_tube_into_rack"))  # not an all_of task


def test_rack_fill_runs_the_tubes_in_order(mj):
    evaluate, make_env = mj
    from skillforge.skills import FillRack

    task = load("franka_fill_tube_rack")
    _, (ep,) = evaluate(make_env, FillRack(), task, [0])
    assert ep.success, (ep.failure, ep.failure_detail)
    # the episode ends the moment all three are seated, which can be before the last release
    assert [e for e in ep.events if e.startswith("grasp:")] == ["grasp:tube_a", "grasp:tube_b", "grasp:tube_c"]
