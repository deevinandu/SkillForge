"""T2: domino tasks and ToppleChain (survey, repair, push, nudge) under disturbances.

The full loop runs here on a kinematic domino world (`tests/domino_kinematic.py`): it checks the
skill's decisions, not physics. The MuJoCo tests at the end are the real check.
"""

import json
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
DOMINO_TASKS = ("franka_topple_domino_chain", "franka_topple_domino_chain_knock", "franka_setup_domino_chain",
                "franka_build_domino_chain_scattered", "franka_rebuild_domino_chain", "franka_rebuild_fallen_dominoes",
                "franka_retopple_domino_chain")


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
    assert t.scene.disturbances or t.scene.scatter  # these tasks are about recovering from them
    if name == "franka_setup_domino_chain":
        start = [np.asarray(o.pose.pos) for o in t.scene.objects]
        assert all(abs(p[1] - lane.center[1]) > lane.half_extents[1] + h / 2 for p in start)  # a fall stays out of the lane
        gaps = [np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(start) for b in start[i + 1 :]]
        assert min(gaps) > h  # too far apart to fall into each other


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


# Measured on MuJoCo (2026-11-28, the 55 cm cell, ToppleChain 0.4, before merging main):
# build_scattered 17/20, rebuild_domino_chain 19/20, rebuild_fallen 23/30, retopple 29/30.
# A majority of 5 seeds keeps CI honest without tripping on the odd failing seed.
MEASURED = DOMINO_TASKS


@pytest.mark.parametrize("name", MEASURED)
def test_mujoco_domino_tasks_mostly_succeed(mj, name):
    report, eps = mj(load(name), range(5))
    ok = sum(e.success for e in eps)
    assert ok >= 3, [(e.seed, e.failure, e.failure_detail) for e in eps if not e.success]


def test_mujoco_ledge_stand_up(mj):
    """LedgeStandUp alone, one flat domino (diag_ledge: 16/16 in MuJoCo)."""
    from skillforge.sim import make_env
    from skillforge.sim.scatter import rest_height, rest_quat
    from skillforge.skills.dominoes import LedgeStandUp, tilt

    d = json.loads(load("franka_rebuild_fallen_dominoes").model_dump_json())
    sc = d["scene"]
    sc["objects"], sc["disturbances"], sc["scatter"] = sc["objects"][:1], [], None
    d["success"]["conditions"] = d["success"]["conditions"][:1]
    sc["regions"] = [r for r in sc["regions"] if r["name"] in ("lane", "spot_domino_1")]
    o = TaskSpec.model_validate(d).scene.object("domino_1")
    sc["objects"][0]["pose"] = {"pos": [0.42, 0.10, rest_height(o, "flat")], "quat": [float(v) for v in rest_quat(o, "flat", 0.0)]}
    t = TaskSpec.model_validate(d)
    env = make_env(t)
    obs = env.reset(0)
    spot = np.array([0.44, 0.09, 0.0])
    job = LedgeStandUp(t, "domino_1", next(f for f in t.scene.fixtures if f.ledge), spot, np.array([1.0, 0, 0]),
                       obs.objects["domino_1"].quat)
    for _ in range(1200):
        obs, _ = env.step(job.act(obs))
        if not job.status.running:
            break
    assert not job.status.failed, job.status.reason
    o = obs.objects["domino_1"]
    assert tilt(o.quat) < 0.15 and np.linalg.norm(o.pos[:2] - spot[:2]) < 0.03


# ---------------------------------------------------------------- standing fallen dominoes up


@pytest.mark.parametrize("face", ["flat", "side"])
def test_standup_finds_a_clear_grasp_at_any_yaw(face):
    from skillforge.sim.scatter import rest_height, rest_quat
    from skillforge.skills.dominoes import StandUp

    t = load("franka_build_domino_chain_scattered")
    o = t.scene.object("domino_1")
    su = StandUp(t, "domino_1", np.array([0.44, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    from skillforge.skills.dominoes import Relay

    # the cell: shelf boards on -y (inner face -0.21), open on +y. On the open side every yaw has
    # a tilted grasp; elsewhere some yaws would lean the forearm over the shelves, and Relay's
    # straight-down pinch (ToppleChain's fallback, which also turns the domino) must have one
    for y_near in (0.06, 0.09, 0.12):
        for deg in range(0, 360, 20):
            q = rest_quat(o, face, math.radians(deg))
            for y_far in (-0.12, -0.06, 0.0, 0.20):
                far = np.array([0.40, y_far, rest_height(o, face)])
                assert su.plan_grasp(far, q) is not None or Relay(t, "domino_1", far, 0.0).grasp(far, q) is not None
            g = su.plan_grasp(np.array([0.40, y_near, rest_height(o, face)]), q)
            assert g is not None, (face, y_near, deg)
            pt, R_h, t_ = g
            assert su._clear(R_h, float(pt[2])) and abs(t_) <= max(su.tilts)


@pytest.mark.parametrize("seed", range(4))
def test_builds_the_chain_from_scattered_dominoes(seed):
    ep, sk = run(load("franka_build_domino_chain_scattered"), seed)
    assert ep.success, (ep.failure_detail, sk.log)
    repaired = {e.split(":")[2].split("#")[0] for e in sk.log if ":repair:" in e}
    assert repaired == set(sk.order)  # every domino started off its spot
    assert not any(":standup:" in e for e in sk.log)


def test_retopples_a_knocked_down_chain():
    """Start: all five are knocked over at step 1, so they lie on each other like the end of a run.
    It only counts once every domino stood again. Dominoes resting on another are first relayed
    to a clear bench point (Relay), then stood up."""
    t = load("franka_retopple_domino_chain")
    wins = []
    for seed in range(6):
        ep, sk = run(t, seed)
        if ep.success:
            assert sum(e.startswith("restood:") for e in ep.events) == 5
            assert sum(":standup:" in e or ":ledge:" in e for e in sk.log) >= 5
            assert any(":relay:" in e and "resting on something" in e for e in sk.log)
        wins.append(ep.success)
    assert sum(wins) >= 5, wins


def test_retopple_does_not_count_the_first_fall():
    """Without standing them up, the knocked-down chain is not a success."""
    t = load("franka_retopple_domino_chain")
    env, sk = DominoKinematicEnv(t), ToppleChain()
    sk.stand_up = False
    ep = run_episode(env, sk, t, 0)
    assert not ep.success and any(e.startswith("fell:") for e in ep.events)


@pytest.mark.parametrize("seed", range(4))
def test_rebuilds_fallen_dominoes(seed):
    t = load("franka_rebuild_fallen_dominoes")
    env, sk = DominoKinematicEnv(t), ToppleChain()
    sk.stand_up = True
    ep = run_episode(env, sk, t, seed)
    assert ep.success, (ep.failure_detail, sk.log)
    stood = {e.split(":")[2].split("#")[0].split(" ")[0] for e in sk.log if ":standup:" in e or ":ledge:" in e}
    assert stood == set(sk.order)  # every domino started lying


@pytest.mark.parametrize("seed", range(4))
def test_rebuilds_after_a_domino_is_moved_elsewhere(seed):
    ep, sk = run(load("franka_rebuild_domino_chain"), seed)
    assert ep.success, (ep.failure_detail, sk.log)
    moved = {e.rsplit(":", 1)[1] for e in ep.events if e.startswith("disturbance:relocate:")}
    assert moved  # the first one always fires
    repaired = {e.split(":")[2].split("#")[0] for e in sk.log if ":repair:" in e}
    assert moved <= repaired


def lying_variant(name: str) -> TaskSpec:
    """The task with lying starts / lay-down relocations (what StandUp is for)."""
    t = load(name)
    sc = t.scene
    upd = {"disturbances": tuple(d.model_copy(update={"faces": ("flat", "side")}) if d.kind == "relocate" else d
                                 for d in sc.disturbances)}
    if sc.scatter is not None:
        upd["scatter"] = sc.scatter.model_copy(update={"faces": ("upright", "flat", "side"), "max_yaw": 3.1416})
    return t.model_copy(update={"scene": sc.model_copy(update=upd)})


def test_lying_domino_is_reported_when_stand_up_is_off():
    t = lying_variant("franka_rebuild_domino_chain")
    env, sk = DominoKinematicEnv(t), ToppleChain()
    sk.stand_up = False
    ep = run_episode(env, sk, t, 0)
    assert not ep.success
    assert any("cannot stand" in e and "stand_up is off" in e for e in sk.log)
    assert ep.failure == "unrecovered_disturbance"


@pytest.mark.parametrize("name", ["franka_build_domino_chain_scattered", "franka_rebuild_domino_chain"])
def test_stand_up_path_when_switched_on(name):
    """With stand_up on, lying starts and lay-downs are stood up and rebuilt (the decisions; the
    MuJoCo pinch is what fails)."""
    t = lying_variant(name)
    stood = 0
    for seed in range(4):
        env, sk = DominoKinematicEnv(t), ToppleChain()
        sk.stand_up = True
        ep = run_episode(env, sk, t, seed)
        assert ep.success, (seed, ep.failure_detail, sk.log)
        stood += sum(":standup:" in e or ":ledge:" in e for e in sk.log)
    assert stood > 0


def test_stand_turn_tries_the_next_pose_when_stuck():
    from skillforge.skills.dominoes import StandTurn, hand_quat
    from skillforge.skills.primitives import Context

    c = Context(obj="domino_1")
    far, near = hand_quat(1.0), hand_quat(0.05)  # stuck at DOWN: 1 rad off the first, 0.05 off the second
    c.bb["stand_cands"] = [far, near]
    node = StandTurn(1.0, rate=0.04, window=5)
    obs = _obs({"domino_1": ((0.4, 0.2, 0.3), (1, 0, 0, 0))})
    status = None
    for _ in range(40):
        _, status = node.tick(obs, c)
        if status.done or status.failed:
            break
    assert status.done and c.bb.get("stand_tries") == 2 and np.allclose(c.quat, near)


# ---------------------------------------------------------------- judging what is feasible (v0.3)


def test_work_points_stay_on_the_table_and_off_the_shelves():
    """The cell (Sumedha, 2026-11-27): table 55 x 90 cm; on one side three shelf boards reaching
    6.5 cm over the table (lowest 6 cm up, 8 cm gaps); on the other a frame of posts."""
    t = load("franka_retopple_domino_chain")
    assert t.scene.table.width == 0.55 and any(f.ledge for f in t.scene.fixtures)
    sk = ToppleChain()
    sk.reset(t)
    shelf_face = max(f.center[1] + f.half_extents[1] for f in t.scene.fixtures if f.name.startswith("shelf_"))
    assert sk.y_lo >= shelf_face + 0.09 and sk.y_hi <= 0.275 - 0.04
    pos = {k: np.array([0.45 + 0.03 * i, 0.0, 0.008]) for i, k in enumerate(sk.order)}
    quat = {k: np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]) for k in sk.order}  # flat
    for k in sk.order:
        spot = sk.standup_spot(k, pos, list(sk.order))
        assert spot is not None and sk.y_lo <= spot[1] <= sk.y_hi
        plan = sk.relay_plan(k, pos, quat)
        assert plan is not None and sk.y_lo <= plan[0][1] <= sk.y_hi


def test_ledge_stand_up_is_tried_first_and_works_in_the_kinematic_world():
    t = load("franka_rebuild_fallen_dominoes")
    for seed in range(3):
        ep, sk = run(t, seed)
        assert ep.success, (seed, ep.failure_detail, sk.log)
        assert any(":ledge:" in e for e in sk.log)


def test_ledge_geometry():
    from skillforge.skills.dominoes import LedgeStandUp

    t = load("franka_rebuild_fallen_dominoes")
    ledge = next(f for f in t.scene.fixtures if f.ledge)
    flat = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])  # long axis along +x
    job = LedgeStandUp(t, "domino_1", ledge, np.array([0.40, 0.09, 0.0]), np.array([1.0, 0.0, 0.0]), flat)
    edge = ledge.center[1] + ledge.half_extents[1]
    assert job.sgn == 1.0 and math.isclose(job.edge_y, edge)
    # centre of mass stays on the board, the +z end overhangs by `overhang`
    assert job.place_xy[1] < edge and math.isclose(job.place_xy[1] + job.h - edge, job.overhang, abs_tol=1e-9)
    assert math.isclose(abs(job.turn), math.pi / 2, abs_tol=1e-6)  # +x -> +y
    from skillforge.skills.grasp import quat_to_mat

    R = quat_to_mat(job.level_quat())
    assert np.allclose(R[:, 2], [0, -1, 0]) and np.allclose(R[:, 1], [0, 0, 1])  # into the board, closing vertically
    for q in job.down_quats():
        R = quat_to_mat(q)
        assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-9) and abs(R[0, 1]) > 0.999  # down, closing along the lane


def _lying_at(t: TaskSpec, name: str, xy) -> TaskSpec:
    objs = []
    for o in t.scene.objects:
        if o.name == name:
            q = (math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0)  # flat, top toward the robot
            o = o.model_copy(update={"pose": o.pose.model_copy(update={"pos": (xy[0], xy[1], o.size[0]), "quat": q})})
        objs.append(o)
    sc = t.scene.model_copy(update={"objects": tuple(objs), "scatter": None, "disturbances": ()})
    return t.model_copy(update={"scene": sc})


def test_gives_up_on_a_domino_out_of_reach_and_still_topples_the_rest():
    t = _lying_at(load("franka_topple_domino_chain"), "domino_3", (0.75, 0.0))
    t = t.model_copy(update={"max_steps": 4000})
    ep, sk = run(t, 0)
    assert any("giving up on domino_3: out of reach" in e for e in sk.log), sk.log
    assert not any(":standup:domino_3" in e or ":relay:domino_3" in e for e in sk.log)
    assert any(":report:" in e and "domino_3" in e for e in sk.log)
    assert sum(":pushed:" in e for e in sk.log) >= 2  # the row has a gap: both halves are pushed
    assert not ep.success


def test_relays_a_domino_that_lies_against_another():
    t = _lying_at(load("franka_topple_domino_chain"), "domino_5", (0.57, 0.02))
    ep, sk = run(t, 0)
    assert any(":relay:domino_5" in e and "next to" in e for e in sk.log), sk.log
    assert ep.success, (ep.failure_detail, sk.log)


def test_stops_rebuilding_when_out_of_time():
    t = load("franka_rebuild_fallen_dominoes").model_copy(update={"max_steps": 1200})
    ep, sk = run(t, 0)
    assert any("out of time" in e for e in sk.log), sk.log
    assert sum(":standup:" in e for e in sk.log) <= 2


def test_a_failed_stand_up_sets_the_domino_down_instead_of_dropping_it(monkeypatch):
    from skillforge.skills.dominoes import StandUp

    # No upright pose once in the air: every stand-up fails there, holding the domino.
    t = load("franka_rebuild_fallen_dominoes").model_copy(update={"max_steps": 3000})
    env, sk = DominoKinematicEnv(t), ToppleChain()
    sk.use_ledge = False  # this is about StandUp
    sk.reset(t)
    orig = StandUp.upright_hands

    def none_in_the_air(self, R_h, R_d, p_e, p_d, k=6, min_sep=0.35):
        return [] if p_e[2] > 0.1 else orig(self, R_h, R_d, p_e, p_d, k, min_sep)

    monkeypatch.setattr(StandUp, "upright_hands", none_in_the_air)
    run_episode(env, sk, t, 0)
    assert any(":setdown:standup failed: no_upright_pose" in e for e in sk.log), sk.log
    assert any(":park:setdown done" in e or ":rehome:setdown done" in e for e in sk.log), sk.log


def test_grasps_turn_the_palm_away_from_the_shelf_boards():
    """MuJoCo, the 55 cm cell: repairs and relays of dominoes near the shelf boards timed out on
    the approach (the palm, +-10.5 cm along the closing axis, ran into the boards)."""
    from skillforge.skills.dominoes import Relay, hand_hits, obstacle_boxes

    t = load("franka_rebuild_fallen_dominoes")
    boxes = obstacle_boxes(t)
    tcp = np.array([0.40, -0.13, 0.029])
    # closing along y (toward the boards) hits them; along x does not
    assert hand_hits(boxes, tcp, 0.0) and not hand_hits(boxes, tcp, math.pi / 2)
    assert not hand_hits(boxes, np.array([0.40, 0.0, 0.029]), 0.0)  # mid-table: either way is fine
    # a flat domino near the boards, long axis along x: Relay closes across its length instead of its width
    flat_x = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])  # long axis along x
    pos = np.array([0.40, -0.13, 0.0079])
    g = Relay(t, "domino_1", pos, 0.0).grasp(pos, flat_x)
    assert g is not None and not hand_hits(boxes, g[0], g[1])


@pytest.mark.parametrize("name", DOMINO_TASKS)
def test_every_starting_domino_has_a_grip_clear_of_the_shelves(name):
    """MuJoCo (2026-11-28): the setup task's staging dominoes at y = -0.18, 3 cm from the shelf
    boards, could not be gripped either way without the hand hitting them (collision, 0/5)."""
    from skillforge.skills.dominoes import hand_hits, obstacle_boxes

    t = load(name)
    boxes = obstacle_boxes(t)
    sc = t.scene.scatter
    scattered = set(sc.objects or [o.name for o in t.scene.objects if not o.fixed]) if sc is not None else set()
    points = [(o.name, o.pose.pos[:2]) for o in t.scene.objects if o.name not in scattered]
    for r in sc.regions if sc is not None else ():  # where scattered dominoes can start: the corners
        c, h = t.scene.region(r).center, t.scene.region(r).half_extents
        points += [(r, (c[0] + sx * h[0], c[1] + sy * h[1])) for sx in (-1, 1) for sy in (-1, 1)]
    for what, xy in points:
        tcp = np.array([xy[0], xy[1], 0.029])
        assert not all(hand_hits(boxes, tcp, yaw) for yaw in (0.0, math.pi / 2)), (name, what)
