import numpy as np
import pytest

from skillforge.contracts import Action, Observation, RandomizationSpec
from skillforge.sim import StubEnv
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    Fallback,
    HoldMonitor,
    MoveTo,
    Node,
    Retry,
    Runner,
    Sequence,
    approach,
    descend_until,
    fail,
    grip,
    lift,
    move_to,
    release,
    retreat,
)


@pytest.fixture
def calm_task(stub_task):
    return stub_task.model_copy(update={"scene": stub_task.scene.model_copy(update={"randomization": RandomizationSpec()})})


def obs_at(ee, width=0.08, objects=None, force=None):
    return Observation(
        t=0.0,
        qpos=np.zeros(3),
        qvel=np.zeros(3),
        ee_pos=np.asarray(ee, dtype=float),
        ee_quat=np.array([0.0, 1, 0, 0]),
        gripper_width=width,
        objects=objects or {},
        contact_force=force,
    )


def run(env, node, ctx, steps=200):
    obs = env._observe()
    runner = Runner(node, ctx)
    for _ in range(steps):
        obs, _ = env.step(runner.act(obs))
        if not runner.status.running:
            break
    return runner, obs


class Scripted(Node):
    """Runs for `n` steps then finishes with `end` (a Status)."""

    def __init__(self, n, end=DONE, name="scripted"):
        self.n, self.end, self.name, self.k, self.resets = n, end, name, 0, 0

    def reset(self):
        self.k = 0
        self.resets += 1

    def tick(self, obs, ctx):
        if self.k >= self.n:
            return None, self.end
        self.k += 1
        return Action(ee_pos=obs.ee_pos), RUNNING


# ---------------------------------------------------------------- phase machine


def test_move_to_arrives_and_times_out(calm_task):
    env = StubEnv(calm_task)
    env.reset(0)
    target = np.array([0.45, 0.05, 0.2])
    runner, obs = run(env, move_to(target, tol=0.005), Context())
    assert runner.status.done and np.linalg.norm(obs.ee_pos - target) < 0.005

    env.reset(0)
    runner, _ = run(env, MoveTo(np.array([2.0, 0, 0.2]), timeout=5, name="far"), Context())
    assert runner.status.failed and runner.status.reason == "timeout:far"


def test_sequence_chains_finished_nodes_in_the_same_step():
    ctx = Context()
    seen = []
    seq = Sequence([Do(lambda o, c: seen.append("a")), Do(lambda o, c: seen.append("b")), Scripted(2)])
    act, st = seq.tick(obs_at([0, 0, 0.2]), ctx)
    assert seen == ["a", "b"] and st == RUNNING and act is not None
    seq.tick(obs_at([0, 0, 0.2]), ctx)
    act, st = seq.tick(obs_at([0, 0, 0.2]), ctx)
    assert st == DONE and act is None


def test_sequence_fails_fast():
    seq = Sequence([Do(lambda o, c: "boom"), Do(lambda o, c: pytest.fail("ran after failure"))])
    _, st = seq.tick(obs_at([0, 0, 0]), Context())
    assert st.failed and st.reason == "boom"


def test_retry_budget_per_reason_prefix():
    child = Scripted(1, end=fail("grasp_miss:x"))
    r = Retry(child, {"grasp_miss": 2, "grasp_slip": 5})
    ctx = Context()
    statuses = []
    for _ in range(10):
        _, st = r.tick(obs_at([0, 0, 0]), ctx)
        statuses.append(st)
        if st.failed:
            break
    assert statuses[-1].failed and statuses[-1].reason == "grasp_miss:x"
    assert ctx.bb["retries"] == ["grasp_miss:x", "grasp_miss:x"]
    assert r.used == {"grasp_miss": 2}

    other = Retry(Scripted(0, end=fail("timeout:x")), {"grasp_miss": 2})
    _, st = other.tick(obs_at([0, 0, 0]), Context())
    assert st.failed and st.reason == "timeout:x"  # not in budget: no retry


def test_retry_int_budget_and_on_retry_hook():
    calls = []
    r = Retry(Scripted(0, end=fail("anything")), 3, on_retry=lambda c, why: calls.append(why))
    _, st = r.tick(obs_at([0, 0, 0]), Context())
    assert st.failed and calls == ["anything"] * 3


def test_fallback_tries_next_on_failure():
    fb = Fallback([Do(lambda o, c: "nope"), Scripted(1)])
    ctx = Context()
    _, st = fb.tick(obs_at([0, 0, 0]), ctx)
    assert st == RUNNING
    _, st = fb.tick(obs_at([0, 0, 0]), ctx)
    assert st == DONE
    _, st = Fallback([Do(lambda o, c: "a"), Do(lambda o, c: "b")]).tick(obs_at([0, 0, 0]), Context())
    assert st.failed and st.reason == "b"


def test_runner_holds_still_after_finishing():
    runner = Runner(Do(lambda o, c: None), Context())
    a = runner.act(obs_at([0.1, 0.2, 0.3]))
    b = runner.act(obs_at([0.1, 0.2, 0.3]))
    assert runner.status.done
    np.testing.assert_allclose(a.ee_pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(b.ee_pos, a.ee_pos)


# ---------------------------------------------------------------- primitives


def test_descend_stops_on_contact_force():
    node = descend_until(np.array([0, 0, 0.0]), contact_force=5.0)
    _, st = node.tick(obs_at([0, 0, 0.1], force=np.array([0, 0, 1.0])), Context())
    assert st == RUNNING
    _, st = node.tick(obs_at([0, 0, 0.05], force=np.array([0, 0, 9.0])), Context())
    assert st == DONE
    # No force sensor: height alone decides.
    _, st = descend_until(np.zeros(3)).tick(obs_at([0, 0, 0.05]), Context())
    assert st == RUNNING


def test_approach_hovers_back_along_gripper_axis():
    node = approach(np.array([0.5, 0, 0.02]), hover=0.1)
    act, st = node.tick(obs_at([0.4, 0, 0.3]), Context())
    assert st == RUNNING
    np.testing.assert_allclose(act.ee_pos, [0.5, 0, 0.12], atol=1e-9)  # DOWN: z axis points -z


def test_hold_monitor_width_collapse_and_drift():
    m = HoldMonitor()
    objs = {"o": type("S", (), {"pos": np.array([0, 0, 0.1])})()}
    m.arm(obs_at([0, 0, 0.1], width=0.04), np.array([0, 0, 0.1]))
    assert m.check(obs_at([0, 0, 0.1], width=0.04, objects=objs), "o") is None
    assert m.check(obs_at([0, 0, 0.1], width=0.0, objects=objs), "o") == "width_collapsed"

    m.arm(obs_at([0, 0, 0.1], width=0.0), np.array([0, 0, 0.1]))  # stub-like: width uninformative
    assert m.check(obs_at([0, 0, 0.1], width=0.0, objects=objs), "o") is None
    reasons = [m.check(obs_at([0, 0, 0.135], width=0.0, objects=objs), "o") for _ in range(3)]
    assert reasons[:2] == [None, None] and reasons[2].startswith("object_drifting")
    assert m.check(obs_at([0, 0, 0.2], width=0.0, objects=objs), "o").startswith("object_left_gripper")


def test_primitives_pick_and_lift_cube_in_stub(calm_task):
    """approach -> descend -> grip -> lift, composed by hand, lifts the cube (true state)."""
    env = StubEnv(calm_task)
    env.reset(0)
    ctx = Context(obj="cube")
    cube = lambda o, c: c.obj_pos
    tree = Sequence([approach(cube), descend_until(cube), grip(), lift(0.15), release(), retreat(0.25)])
    runner, obs = run(env, tree, ctx)
    assert runner.status.done, (runner.status, ctx.log)
    names = [e.split(":", 2)[2] for e in ctx.log if ":done:" in e]
    assert names == ["approach", "descend", "grip", "lift_up", "lift_verify", "lift", "release", "retreat"]
    assert obs.ee_pos[2] == pytest.approx(0.25, abs=0.01)


def test_missed_grasp_detected_at_grip_or_on_lift(calm_task):
    # Grip 5 cm beside the cube: the stub closes on nothing and the cube stays on the table.
    beside = lambda o, c: c.obj_pos + [0.05, 0, 0]
    env = StubEnv(calm_task)

    env.reset(0)
    ctx = Context(obj="cube")
    runner, _ = run(env, Sequence([approach(beside), descend_until(beside), grip(), lift(0.15)]), ctx)
    assert runner.status.failed and runner.status.reason.startswith("grasp_miss:empty")

    env.reset(0)
    ctx = Context(obj="cube")
    tree = Sequence([approach(beside), descend_until(beside), grip(miss_dist=0.2), lift(0.15)])
    runner, _ = run(env, tree, ctx)  # grip check disabled: the lift catches it
    assert runner.status.failed and runner.status.reason.startswith("grasp_miss:not_lifted")
