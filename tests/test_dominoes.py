"""T2: domino tasks and ToppleChain (survey, repair, push, nudge) under disturbances.

The full loop runs here on a kinematic domino world (`tests/domino_kinematic.py`): it checks the
skill's decisions, not physics. The MuJoCo tests at the end are the real check.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from domino_kinematic import DominoKinematicEnv

from skillforge.contracts import ObjectState, Observation, Skill, TaskSpec, success_conditions
from skillforge.eval import run_episode
from skillforge.eval.regression import library_sources
from skillforge.skills import ToppleChain, load_skill
from skillforge.skills.dominoes import hand_quat, heading_x, nearest_turn, tilt, wrap_half_pi
from skillforge.skills.grasp import DOWN, quat_angle

TASKS = Path(__file__).resolve().parents[1] / "tasks"
DOMINO_TASKS = ("franka_topple_domino_chain", "franka_topple_domino_chain_knock", "franka_setup_domino_chain")


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def without_disturbances(t: TaskSpec) -> TaskSpec:
    return t.model_copy(update={"scene": t.scene.model_copy(update={"disturbances": ()})})


# ---------------------------------------------------------------- tasks


@pytest.mark.parametrize("name", DOMINO_TASKS)
def test_domino_task_geometry(name):
    t = load(name)
    conds = success_conditions(t.success)
    lane = t.scene.region("lane")
    spots = [np.asarray(t.scene.region(f"spot_{c.object}").center) for c in conds]
    xs = sorted(p[0] for p in spots)
    h = 2 * t.scene.object(conds[0].object).size[2]
    gaps = np.diff(xs)
    assert np.all(gaps > 4 * t.scene.object(conds[0].object).size[0]) and np.all(gaps < 0.75 * h)  # a chain
    for p in spots:  # every spot, and the last domino lying flat past it, is inside the lane
        assert np.all(np.abs(p - lane.center) <= lane.half_extents)
    assert xs[-1] + h <= lane.center[0] + lane.half_extents[0]
    assert all(c.direction == (1.0, 0.0) and c.region == "lane" for c in conds)
    assert t.scene.disturbances  # these tasks are about recovering from them
    if name == "franka_setup_domino_chain":
        start = sorted(o.pose.pos for o in t.scene.objects)
        assert all(abs(p[1] - lane.center[1]) > lane.half_extents[1] for p in start)  # staged outside the lane
        assert np.all(np.diff([p[0] for p in start]) > h)  # too far apart to fall into each other


def test_library_skills_reject_domino_tasks():
    """Every other library skill refuses a domino task in reset (ValueError, or for the older
    leaf-only skills an AttributeError that the harness records as an exception), so no baseline
    is written for the pair."""
    t = load("franka_topple_domino_chain")
    for name, src in library_sources().items():
        sk = load_skill(src)
        if name == "topple_chain":
            sk.reset(t)
            continue
        with pytest.raises((ValueError, AttributeError)):
            sk.reset(t)


def test_topple_chain_setup():
    sk = ToppleChain()
    assert isinstance(sk, Skill)
    sk.reset(load("franka_topple_domino_chain"))
    assert sk.order == [f"domino_{i}" for i in range(1, 6)] and sk.yaw == 0.0
    np.testing.assert_allclose(sk.d, [1, 0, 0])
    with pytest.raises(ValueError, match="object_toppled"):
        ToppleChain().reset(load("franka_stack_block_pyramid"))
    t = load("franka_topple_domino_chain")
    no_spots = t.model_copy(update={"scene": t.scene.model_copy(update={"regions": t.scene.regions[:1]})})
    with pytest.raises(ValueError, match="spot_domino_1"):
        ToppleChain().reset(no_spots)


def test_angle_helpers():
    assert wrap_half_pi(math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_half_pi(-math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_half_pi(2.0) == pytest.approx(2.0 - math.pi)
    assert nearest_turn(0.0, 3.0, math.pi) == pytest.approx(math.pi)
    assert quat_angle(hand_quat(0.0), DOWN) < 1e-9
    assert heading_x(hand_quat(0.7)) == pytest.approx(0.7)


def _obs(objs: dict[str, tuple]) -> Observation:
    o = {k: ObjectState(pos=np.asarray(p, dtype=float), quat=np.asarray(q, dtype=float)) for k, (p, q) in objs.items()}
    return Observation(t=0, qpos=np.zeros(7), qvel=np.zeros(7), ee_pos=np.zeros(3), ee_quat=DOWN, gripper_width=0.08, objects=o)


def test_classify():
    t = load("franka_topple_domino_chain")
    sk = ToppleChain()
    sk.reset(t)
    up = (1.0, 0, 0, 0)
    objs = {k: (sk.spots[k], up) for k in sk.order}
    yawed = (math.cos(0.1), 0, 0, math.sin(0.1))  # 0.2 rad
    fwd = (math.cos(0.6), 0, math.sin(0.6), 0)  # tipped 1.2 rad about +y: top toward +x
    side = (math.cos(0.6), -math.sin(0.6), 0, 0)  # about -x: top toward +y
    objs["domino_2"] = (sk.spots["domino_2"] + [0, 0.012, 0], up)
    objs["domino_3"] = (sk.spots["domino_3"], yawed)
    objs["domino_4"] = (sk.spots["domino_4"], fwd)
    objs["domino_5"] = (sk.spots["domino_5"], side)
    st = sk.classify([_obs(objs)])
    assert st == {"domino_1": "ok", "domino_2": "misplaced", "domino_3": "misplaced", "domino_4": "down", "domino_5": "down_bad"}
    assert tilt(np.array(fwd)) == pytest.approx(1.2)


# ---------------------------------------------------------------- the loop, on the kinematic world


def run(task: TaskSpec, seed: int):
    env, sk = DominoKinematicEnv(task), ToppleChain()
    ep = run_episode(env, sk, task, seed)
    return ep, sk


def test_pushes_an_undisturbed_chain_over():
    ep, sk = run(without_disturbances(load("franka_topple_domino_chain")), 0)
    assert ep.success, (ep.failure_detail, sk.log)
    assert not any(":repair:" in e for e in sk.log) and any(":push:domino_1" in e for e in sk.log)


@pytest.mark.parametrize("seed", range(6))
def test_repairs_bumped_dominoes_then_topples(seed):
    t = load("franka_topple_domino_chain")
    ep, sk = run(t, seed)
    assert ep.success, (ep.failure_detail, sk.log)
    bumped = [e for e in ep.events if e.startswith("disturbance:pose:")]
    assert bumped, ep.events  # the first bump always fires
    repaired = {e.split(":")[2].split("#")[0] for e in sk.log if ":repair:" in e}
    assert {e.rsplit(":", 1)[1] for e in bumped} <= repaired


@pytest.mark.parametrize("seed", range(4))
def test_nudges_what_an_early_knock_left_standing(seed):
    t = load("franka_topple_domino_chain_knock")
    ep, sk = run(t, seed)
    assert ep.success, (ep.failure_detail, sk.log)
    assert any(e.startswith("disturbance:knock:") for e in ep.events)


@pytest.mark.parametrize("seed", range(3))
def test_sets_up_the_chain_from_the_side(seed):
    t = load("franka_setup_domino_chain")
    ep, sk = run(t, seed)
    assert ep.success, (ep.failure_detail, sk.log)
    repaired = [e.split(":")[2].split("#")[0] for e in sk.log if ":repair:" in e]
    assert list(dict.fromkeys(repaired)) == ["domino_4", "domino_3", "domino_2", "domino_1"]  # farthest first
    bumped = {e.rsplit(":", 1)[1] for e in ep.events if e.startswith("disturbance:pose:")}
    assert bumped <= set(repaired)


# ---------------------------------------------------------------- MuJoCo


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.eval import evaluate
    from skillforge.sim import make_env
    from skillforge.sim.assets import AssetError

    def ev(task, seeds):
        try:
            return evaluate(make_env, ToppleChain(), task, seeds)
        except AssetError as e:
            pytest.skip(f"menagerie not available: {e}")

    return ev


def test_mujoco_undisturbed_chain(mj):
    _, (ep,) = mj(without_disturbances(load("franka_topple_domino_chain")), [0])
    assert ep.success, (ep.failure, ep.failure_detail)


@pytest.mark.parametrize("name", DOMINO_TASKS)
def test_mujoco_domino_tasks_seed0(mj, name):
    _, (ep,) = mj(load(name), [0])
    assert ep.success, (ep.failure, ep.failure_detail)
