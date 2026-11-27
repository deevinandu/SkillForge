"""T1: the object_toppled predicate and mid-episode disturbances (schedule, StubEnv, MujocoEnv)."""

import json
import math

import numpy as np
import pytest

from skillforge.contracts import Action, ObjectToppled, TaskSpec
from skillforge.sim import StubEnv, is_success
from skillforge.sim.disturbances import MAX_WAIT, DisturbanceSchedule, knock_axis
from skillforge.sim.success import tilt_of, up_axis


def tilted(angle: float, heading: float = 0.0) -> np.ndarray:
    """Quaternion of an upright body tipped by `angle` so its top moves toward `heading`."""
    ax = knock_axis(heading)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * ax])


def domino_task(source="stub", n=3, disturbances=(), **succ) -> TaskSpec:
    robot = {"name": "r", "source": source}
    if source != "stub":
        robot.update(name="franka_panda", ee_site="tcp", gripper="parallel", max_gripper_width=0.08)
    dom = {"shape": "box", "size": [0.005, 0.02, 0.04], "mass": 0.03}
    return TaskSpec.model_validate({
        "id": "dominoes_t1",
        "instruction": "topple the dominoes",
        "scene": {
            "robot": robot,
            "objects": [{**dom, "name": f"d{i}", "pose": {"pos": [0.42 + 0.05 * i, 0.0, 0.04]}} for i in range(n)],
            "regions": [{"name": "lane", "center": [0.5, 0, 0.03], "half_extents": [0.2, 0.06, 0.05]}],
            "disturbances": list(disturbances),
        },
        "success": {"kind": "all_of", "conditions": [
            {"kind": "object_toppled", "object": f"d{i}", "direction": [1, 0], "region": "lane", **succ}
            for i in range(n)
        ]},
        "max_steps": 200,
    })


# ---------------------------------------------------------------- predicate


def test_up_axis_and_tilt():
    np.testing.assert_allclose(up_axis(np.array([1.0, 0, 0, 0])), [0, 0, 1])
    q = tilted(0.7, heading=0.0)
    z = up_axis(q)
    assert z[0] > 0 and abs(z[1]) < 1e-9  # top moved toward +x
    assert tilt_of(q) == pytest.approx(0.7)


def test_toppled_predicate():
    t = domino_task(n=1)
    pos = {"d0": np.array([0.45, 0.0, 0.03])}
    ok = lambda q: is_success(t, pos, pos, quats={"d0": q})
    assert not ok(np.array([1.0, 0, 0, 0]))  # upright
    assert not ok(tilted(0.4))  # leaning, under min_tilt
    assert ok(tilted(0.7)) and ok(tilted(math.pi / 2))  # toward +x
    assert ok(tilted(1.0, heading=0.6)) and not ok(tilted(1.0, heading=1.0))  # dir_tol 0.8
    assert not ok(tilted(1.2, heading=math.pi))  # fell backwards
    out = {"d0": np.array([0.45, 0.2, 0.0])}
    assert not is_success(t, out, out, quats={"d0": tilted(1.2)})  # outside the lane
    with pytest.raises(ValueError, match="quats"):
        is_success(t, pos, pos)
    loose = ObjectToppled(object="d0")  # no direction, no region: any fall counts
    t2 = t.model_copy(update={"success": loose})
    assert is_success(t2, out, out, quats={"d0": tilted(1.2, heading=2.0)})


# ---------------------------------------------------------------- schedule


POSE = {"kind": "pose", "objects": ["d1"], "at_step": {"lo": 5, "hi": 30}, "xy": {"lo": 0.01, "hi": 0.03},
        "yaw": {"lo": 0.1, "hi": 0.4}}


def test_schedule_is_seeded_and_logged():
    t = domino_task(disturbances=[POSE])
    a, b, c = (DisturbanceSchedule(t, s) for s in (3, 3, 4))
    assert a.sample == b.sample and a.sample != c.sample
    (p,) = a.plans
    assert p.obj == "d1" and 5 <= p.due <= 30 and p.due == p.delay
    assert 0.01 <= math.hypot(p.dx, p.dy) <= 0.03 + 1e-12 and 0.1 <= abs(p.dyaw) <= 0.4
    assert a.sample["disturb0.0.obj"] == 1.0 and a.sample["disturb0.0.fires"] == 1.0
    assert a.update(p.due - 1, []) == []
    assert a.update(p.due, []) == [p]
    assert a.applied(p) == "disturbance:pose:d1" and a.update(p.due + 1, []) == []


def test_schedule_prob_before_event_count():
    never = domino_task(disturbances=[{**POSE, "prob": 0.0}])
    s = DisturbanceSchedule(never, 0)
    assert s.update(100, []) == [] and s.plans[0].state == "cancelled"
    late = domino_task(disturbances=[{**POSE, "at_step": {"lo": 50, "hi": 50}, "before_event": "toppled:"}])
    s = DisturbanceSchedule(late, 0)
    s.update(10, ["toppled:d0"])
    assert s.update(50, []) == [] and s.plans[0].state == "cancelled"
    many = domino_task(disturbances=[{**POSE, "objects": [], "count": 2}])
    s = DisturbanceSchedule(many, 0)
    assert len(s.plans) == 2 and len({p.obj for p in s.plans}) == 2


def test_schedule_after_event_targets_the_event_object():
    t = domino_task(disturbances=[{**POSE, "objects": [], "after_event": "release:", "target": "event_object",
                                   "at_step": {"lo": 2, "hi": 2}}])
    s = DisturbanceSchedule(t, 0)
    (p,) = s.plans
    assert p.obj is None and p.due is None and s.update(1, ["grasp:d2"]) == []
    assert s.update(7, ["release:d2"]) == [] and p.obj == "d2" and p.due == 9
    assert s.update(9, []) == [p] and s.sample["disturb0.0.obj"] == 2.0
    for k in range(MAX_WAIT):
        assert s.blocked(p) is None, k
    assert s.blocked(p) == "disturbance_skipped:d2" and s.update(10, []) == []


def test_no_disturbances_no_new_sample_keys(stub_task):
    env = StubEnv(stub_task)
    env.reset(0)
    assert not any(k.startswith("disturb") for k in env.randomization_sample())


# ---------------------------------------------------------------- StubEnv


def test_stub_applies_pose_disturbance_deterministically():
    t = domino_task(disturbances=[POSE])
    env = StubEnv(t)
    runs = []
    for _ in range(2):
        obs0 = env.reset(7)
        due = int(env.randomization_sample()["disturb0.0.step"])
        park = Action(ee_pos=np.array([0.3, 0.3, 0.3]))
        hits = []
        for k in range(1, 40):
            obs, info = env.step(park)
            if info.events:
                hits.append((k, info.events))
        runs.append((due, hits, obs.objects["d1"].pos - obs0.objects["d1"].pos, obs.objects["d1"].quat))
    (due, hits, dp, q), again = runs
    assert hits == [(max(1, due), ["disturbance:pose:d1"])]
    assert 0.01 <= np.linalg.norm(dp[:2]) <= 0.03 + 1e-9 and tilt_of(q) == pytest.approx(0.0, abs=1e-9)
    assert abs(2 * math.atan2(q[3], q[0])) >= 0.1 - 1e-9  # yawed
    assert again[0] == due and again[1] == hits and np.allclose(again[2], dp)


def test_stub_knock_topples_and_succeeds():
    knock = {"kind": "knock", "objects": [], "count": 3, "at_step": {"lo": 1, "hi": 3},
             "ang_vel": {"lo": 3, "hi": 3}, "knock_dir": [1, 0]}
    env = StubEnv(domino_task(disturbances=[knock]))
    env.reset(0)
    info = None
    for _ in range(5):
        _, info = env.step(Action(ee_pos=np.array([0.3, 0.3, 0.3])))
        if info.success:
            break
    assert info.success


# ---------------------------------------------------------------- MujocoEnv


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.sim.assets import AssetError
    from skillforge.sim.mujoco_env import MujocoEnv

    def make(task):
        try:
            return MujocoEnv(task)
        except AssetError as e:
            pytest.skip(f"menagerie not available: {e}")

    return make


def _park(env):
    obs = env._observe()
    return Action(ee_pos=obs.ee_pos.copy(), ee_quat=obs.ee_quat.copy())


def test_mujoco_dominoes_stand_still(mj):
    env = mj(domino_task(source="menagerie:franka_emika_panda"))
    obs = env.reset(0)
    a = _park(env)
    for _ in range(40):
        obs, info = env.step(a)
        assert not info.success and not any(e.startswith("toppled:") for e in info.events)
    for o in obs.objects.values():
        assert tilt_of(o.quat) < 0.02 and abs(o.pos[2] - 0.04) < 0.002


def test_mujoco_knock_topples_the_chain(mj):
    knock = {"kind": "knock", "objects": ["d0"], "at_step": {"lo": 5, "hi": 5},
             "ang_vel": {"lo": 4, "hi": 4}, "knock_dir": [1, 0]}
    env = mj(domino_task(source="menagerie:franka_emika_panda", disturbances=[knock]))
    env.reset(0)
    a = _park(env)
    events, success = [], False
    for _ in range(120):
        _, info = env.step(a)
        events += info.events
        if info.success:
            success = True
            break
    assert "disturbance:knock:d0" in events
    assert [e for e in events if e.startswith("toppled:")][:1] == ["toppled:d0"]
    assert success, events


def test_mujoco_pose_disturbance_is_reproducible(mj):
    env = mj(domino_task(source="menagerie:franka_emika_panda", disturbances=[POSE]))
    finals = []
    for _ in range(2):
        obs0 = env.reset(11)
        a = _park(env)
        evs = []
        for _ in range(40):
            obs, info = env.step(a)
            evs += info.events
        assert "disturbance:pose:d1" in evs
        assert tilt_of(obs.objects["d1"].quat) < 0.05  # moved, still standing
        finals.append(obs.objects["d1"].pos - obs0.objects["d1"].pos)
    assert 0.005 <= np.linalg.norm(finals[0][:2]) <= 0.031
    np.testing.assert_allclose(finals[0], finals[1], atol=1e-9)


def test_mujoco_existing_task_unchanged_by_disturbance_support(mj):
    from pathlib import Path

    t = TaskSpec.model_validate(json.loads((Path(__file__).parents[1] / "tasks" / "franka_lift_cube.json").read_text()))
    env = mj(t)
    env.reset(3)
    assert not any(k.startswith("disturb") for k in env.randomization_sample())
