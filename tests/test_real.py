"""T7: RealEnv, safety and the fakes. No hardware and no ROS; the MuJoCo tests skip without it."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Action, Range, SimEnv, TaskSpec
from skillforge.eval import evaluate
from skillforge.real import (
    FakeCell,
    FakeRobotIO,
    GuardedSkill,
    PointController,
    RealEnv,
    RobotIO,
    SafetyLimits,
    Watchdog,
    Workspace,
    draw_placement,
    make_fake_cell_env,
)
from skillforge.skills import PickPlace

TASKS = Path(__file__).parent.parent / "tasks"
DT = 0.05


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate(json.loads((TASKS / f"{name}.json").read_text()))


def cell_env(task, *, limits=None, **kw):
    """RealEnv over a FakeCell, returning the cell too."""
    limits = limits or SafetyLimits()
    cell = FakeCell(task)
    env = RealEnv(task, cell, estimator=cell.estimator(), controller=PointController(DT, max_ee_speed=limits.ee_speed),
                  operator=cell, limits=limits, **kw)
    return env, cell


def go(pos, grip=0.0):
    return Action(mode="ee", ee_pos=np.asarray(pos, dtype=float), gripper=grip)


# ---------------------------------------------------------------- the env


def test_real_env_satisfies_the_contracts(stub_task):
    env, cell = cell_env(stub_task)
    assert isinstance(env, SimEnv) and isinstance(cell, RobotIO)
    obs = env.reset(3)
    assert set(obs.objects) == {"cube"} and obs.qpos.shape == (3,)
    assert env.render() is None
    env.close()


def test_a_skill_runs_end_to_end_through_the_harness(stub_task):
    rep, eps = evaluate(make_fake_cell_env, PickPlace(), stub_task, range(10))
    assert rep.n_success == 10
    assert all(any(e.startswith("grasp:") for e in ep.events) for ep in eps)
    assert not any(e.startswith("safety:") for ep in eps for e in ep.events)


def test_same_seed_same_episode(stub_task):
    def run():
        _, eps = evaluate(make_fake_cell_env, PickPlace(), stub_task, [4])
        return eps[0].n_steps, eps[0].events, eps[0].randomization

    assert run() == run()


def test_latency_and_a_soft_arm_slow_the_skill_down_but_do_not_break_it(stub_task):
    crisp, _ = evaluate(make_fake_cell_env, PickPlace(), stub_task, range(5))
    soft, eps = evaluate(partial(make_fake_cell_env, latency_steps=3, gain=0.5), PickPlace(), stub_task, range(5))
    assert crisp.n_success == soft.n_success == 5


def test_fake_robot_tracks_with_latency_and_gain():
    r = FakeRobotIO(np.zeros(2), latency_steps=2, gain=0.5)
    for _ in range(2):
        r.send_joint_targets(np.ones(2), 0.0)
    assert np.allclose(r.read_state().q, 0.0)  # nothing has arrived yet
    r.send_joint_targets(np.ones(2), 0.0)
    assert np.allclose(r.read_state().q, 0.5)
    r.send_joint_targets(np.ones(2), 1.0)
    s = r.read_state()
    assert np.allclose(s.q, 0.75) and np.allclose(s.qvel, 0.25 / r.dt)
    r.stop()
    r.send_joint_targets(np.zeros(2), 0.0)
    assert np.allclose(r.read_state().q, 0.75)  # stopped: targets are ignored


def test_without_an_estimator_there_are_no_objects_and_no_success(stub_task):
    cell = FakeCell(stub_task)
    env = RealEnv(stub_task, cell, controller=PointController(DT))
    obs = env.reset(0)
    assert obs.objects == {}
    obs, info = env.step(go(obs.ee_pos))
    assert obs.objects == {} and not info.success and not info.terminated


def test_an_estimator_that_needs_frames_needs_a_frame_source(stub_task):
    from skillforge.perception import DepthEstimator

    with pytest.raises(ValueError, match="frames"):
        RealEnv(stub_task, FakeCell(stub_task), estimator=DepthEstimator(), controller=PointController(DT))


# ---------------------------------------------------------------- reset and placement


def test_reset_asks_the_operator_for_the_seeds_placement_and_records_what_was_measured(stub_task):
    cell = FakeCell(stub_task, place_error=(0.004, -0.003))
    env = RealEnv(stub_task, cell, estimator=cell.estimator(), controller=PointController(DT), operator=cell)
    obs = env.reset(7)
    sample, targets = draw_placement(stub_task, 7)
    assert env.targets == targets
    assert np.allclose(obs.objects["cube"].pos[:2], np.array(targets["cube"].pos[:2]) + [0.004, -0.003])
    s = env.randomization_sample()
    assert s["cube.dx"] == sample["cube.dx"] and abs(s["cube.place_error"] - 0.005) < 1e-9
    assert abs(s["cube.placed_dx"] - (sample["cube.dx"] + 0.004)) < 1e-9
    # hardware cannot draw physical parameters: reported at nominal, and flagged
    assert s["real.physical_nominal"] == 1.0 and s["friction_scale"] == 1.0 and s["action_latency_steps"] == 0.0


def test_success_is_the_estimators_opinion_of_the_measured_start(stub_task):
    env, cell = cell_env(stub_task)
    env.reset(0)
    bin_centre = np.array(stub_task.scene.region("bin").center)
    cell._obj["cube"][:] = bin_centre  # someone moves the cube into the bin
    _, info = env.step(go(cell.q))
    assert info.success


# ---------------------------------------------------------------- safety


def test_workspace_comes_from_the_tasks_objects_and_regions(stub_task):
    w = Workspace.from_task(stub_task, margin=0.1)
    # cube at (0.5, -0.1) with 5 cm jitter; bin at (0.5, 0.2) +- 5 cm
    assert np.allclose(w.lo, (0.35, -0.25, 0.005)) and np.allclose(w.hi, (0.65, 0.35, 0.6))
    assert w.contains([0.5, 0.0, 0.1]) and not w.contains([0.5, 0.0, 0.0])
    wide = Workspace.from_task(stub_task, include=[np.array([0.2, 0.0, 0.9])])
    assert wide.contains([0.2, 0.0, 0.9])


def test_a_target_outside_the_workspace_is_clamped_and_reported_once(stub_task):
    env, cell = cell_env(stub_task, limits=SafetyLimits(ee_speed=5.0, joint_speed=1e3))
    env.reset(0)
    events = []
    for _ in range(5):
        _, info = env.step(go([2.0, 0.0, -1.0]))
        events += info.events
    assert events == ["safety:workspace"]
    assert all(env.safety.workspace.contains(q) for q, _ in cell.sent)
    assert np.allclose(cell.q, [env.safety.workspace.hi[0], 0.0, env.safety.workspace.lo[2]])
    _, info = env.step(go([0.5, 0.0, 0.2]))  # back inside clears the streak
    _, info = env.step(go([2.0, 0.0, 0.2]))
    assert info.events == ["safety:workspace"]
    assert [e for _, e in env.safety_log] == ["safety:workspace", "safety:workspace"]


def test_the_tool_never_moves_faster_than_the_speed_limit(stub_task):
    env, cell = cell_env(stub_task, limits=SafetyLimits(ee_speed=0.1, joint_speed=1e3))
    obs = env.reset(0)
    start = obs.ee_pos.copy()
    for _ in range(10):
        prev = obs.ee_pos
        obs, _ = env.step(go(start + [0.0, 0.3, 0.0]))
        assert np.linalg.norm(obs.ee_pos - prev) <= 0.1 * DT + 1e-12
    assert abs(np.linalg.norm(obs.ee_pos - start) - 10 * 0.1 * DT) < 1e-9


def test_a_joint_target_too_far_away_is_approached_at_the_joint_speed_limit(stub_task):
    env, cell = cell_env(stub_task, limits=SafetyLimits(joint_speed=0.2))
    obs = env.reset(0)
    target = obs.qpos + [0.1, -0.05, 0.0]
    obs, info = env.step(Action(mode="joint", joint_targets=target))
    assert info.events == ["safety:joint_speed"]
    moved = obs.qpos - cell._home
    assert np.allclose(moved, [0.2 * DT, -0.1 * DT, 0.0])  # the whole step is scaled: direction kept
    for _ in range(20):
        obs, info = env.step(Action(mode="joint", joint_targets=target))
    assert np.allclose(obs.qpos, target)


def test_a_joint_target_outside_the_workspace_is_refused(stub_task):
    env, cell = cell_env(stub_task)
    obs = env.reset(0)
    obs2, info = env.step(Action(mode="joint", joint_targets=np.array([3.0, 0.0, 0.2]), gripper=1.0))
    assert info.events[0] == "safety:workspace" and np.allclose(obs2.qpos, obs.qpos)
    assert cell.sent[-1][1] == 1.0  # the gripper command still goes through


def test_too_much_force_stops_the_arm_for_the_rest_of_the_episode(stub_task):
    env, cell = cell_env(stub_task, limits=SafetyLimits(force=20.0))
    obs = env.reset(0)
    obs, info = env.step(go(obs.ee_pos + [0.0, 0.0, -0.01]))
    assert not info.terminated
    cell.force = np.array([0.0, 0.0, 25.0])
    obs, info = env.step(go(obs.ee_pos + [0.0, 0.0, -0.01]))
    assert info.events == ["safety:force"] and info.terminated and cell.stopped and env.stopped == "force"
    held = cell.q.copy()
    obs, info = env.step(go(obs.ee_pos + [0.0, 0.1, 0.0]))
    assert info.terminated and not info.success and np.allclose(cell.q, held)
    cell.force = None
    env.reset(1)  # homing clears the stop
    assert env.stopped is None and not cell.stopped and env.safety_log == []


class Clock:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_realtime_paces_the_loop_and_the_watchdog_reports_a_late_action(stub_task):
    clock = Clock()
    env, cell = cell_env(stub_task, realtime=True, clock=clock, sleep=clock.sleep)
    obs = env.reset(0)
    for _ in range(3):
        clock.now += 0.01  # the skill and the estimator take 10 ms
        obs, info = env.step(go(obs.ee_pos))
        assert info.events == []
    assert np.allclose(clock.slept, DT - 0.01) and abs(clock.now - (100.0 + 3 * DT)) < 1e-9
    clock.now += 2.5 * DT  # stuck for more than two control periods
    obs, info = env.step(go(obs.ee_pos))
    assert info.events == ["safety:watchdog"]


def test_watchdog_alone():
    clock = Clock()
    w = Watchdog(0.1, clock)
    assert not w.expired()  # never fed: nothing to hold yet
    w.feed()
    clock.now += 0.09
    assert not w.expired()
    clock.now += 0.02
    assert w.expired()


class Boom:
    needs_frames = False

    def __init__(self, after):
        self.n, self.after = 0, after

    def reset(self, task):
        pass

    def estimate(self, frames, obs):
        self.n += 1
        if self.n > self.after:
            raise RuntimeError("camera unplugged")
        return {}


def test_an_exception_in_the_estimator_stops_the_arm(stub_task):
    cell = FakeCell(stub_task)
    env = RealEnv(stub_task, cell, estimator=Boom(after=2), controller=PointController(DT), operator=cell)
    obs = env.reset(0)
    obs, info = env.step(go(obs.ee_pos))
    assert not info.terminated
    obs, info = env.step(go(obs.ee_pos))
    assert info.terminated and info.events == ["safety:exception:RuntimeError"] and cell.stopped
    assert env.stopped == "exception:RuntimeError" and obs is not None


class Crashing(PickPlace):
    def act(self, obs):
        if obs.t > 0.5:
            raise ZeroDivisionError("agent-written bug")
        return super().act(obs)


def test_an_exception_in_the_skill_stops_the_arm_before_the_harness_sees_it(stub_task):
    env, cell = cell_env(stub_task)
    rep, eps = evaluate(lambda task: env, GuardedSkill(Crashing(), env), stub_task, [0])
    assert eps[0].failure == "exception" and eps[0].skill == "pick_place"
    assert cell.stopped and env.stopped == "exception:ZeroDivisionError"


# ---------------------------------------------------------------- MuJoCo: the controller and the dry run


@pytest.fixture(scope="module")
def cube_task():
    pytest.importorskip("mujoco")
    from skillforge.sim.assets import AssetError, resolve_robot_source

    try:
        resolve_robot_source("menagerie:franka_emika_panda")
    except (AssetError, OSError) as e:
        pytest.skip(f"menagerie assets unavailable: {e}")
    task = load("franka_pick_cube_to_bin_hard")
    # the dry run sees true poses, so compare against a sim with no observation noise
    rnd = task.scene.randomization.model_copy(update={"obs_noise_std": Range.fixed(0.0)})
    return task.model_copy(update={"scene": task.scene.model_copy(update={"randomization": rnd})})


def test_a_seed_means_the_same_placement_as_in_sim(cube_task):
    from skillforge.sim import make_env

    env = make_env(cube_task)
    for seed in (0, 5, 123):
        obs = env.reset(seed)
        sample, targets = draw_placement(cube_task, seed)
        assert sample == env.randomization_sample()
        assert np.allclose(obs.objects["cube"].pos[:2], targets["cube"].pos[:2], atol=1e-9)
        assert abs(abs(np.dot(obs.objects["cube"].quat, targets["cube"].quat)) - 1.0) < 1e-6
    env.close()


def test_mujoco_controller_is_the_sims_controller(cube_task):
    from skillforge.real import MujocoController
    from skillforge.sim import make_env

    env = make_env(cube_task)
    obs = env.reset(0)
    c = MujocoController(cube_task, max_ee_speed=0.1)
    pos, quat = c.fk(obs.qpos)
    assert np.allclose(pos, obs.ee_pos, atol=1e-9) and np.allclose(quat, obs.ee_quat, atol=1e-9)
    c.reset(obs.qpos)
    q = obs.qpos
    for _ in range(10):
        q, events = c.solve(Action(mode="ee", ee_pos=pos + [0.0, 0.2, 0.0], ee_quat=quat))
        assert events == []
    moved, _ = c.fk(q)
    assert np.allclose(moved - pos, [0.0, 10 * 0.1 * DT, 0.0], atol=2e-3)  # 0.1 m/s for 10 steps
    env.close()


def test_a_dry_run_at_sim_speed_matches_the_sim_run(cube_task):
    from skillforge.real import make_sim_speed_twin_env
    from skillforge.sim import make_env

    seeds = [0, 1, 2]
    _, sim = evaluate(make_env, PickPlace(), cube_task, seeds)
    _, dry = evaluate(make_sim_speed_twin_env, PickPlace(), cube_task, seeds)
    for a, b in zip(sim, dry, strict=True):
        assert a.success and b.success
        # not step-identical: the sim reports the tool pose one physics substep stale, the
        # real env computes it from the measured joints
        assert abs(a.n_steps - b.n_steps) <= 2
        assert sorted(a.events) == sorted(b.events)
        assert b.randomization["real.dry_run"] == 1.0
        assert {k: v for k, v in b.randomization.items() if k in a.randomization} == a.randomization


def test_a_dry_run_at_the_real_speed_limit_is_slower_and_stays_inside_the_limits(cube_task):
    from skillforge.real import make_twin_env

    env = make_twin_env(cube_task)  # 0.1 m/s
    skill = PickPlace()
    obs = env.reset(0)
    skill.reset(cube_task)
    fastest = 0.0
    for _ in range(cube_task.max_steps):
        prev = obs.ee_pos
        obs, info = env.step(skill.act(obs))
        fastest = max(fastest, float(np.linalg.norm(obs.ee_pos - prev)) / DT)
        assert env.safety.workspace.contains(obs.ee_pos + [0.0, 0.0, 0.01])  # 1 cm slack for tracking error
        if info.success or info.terminated:
            break
    assert fastest < 0.15  # commanded 0.1 m/s; the measured tool lags and catches up
    assert info.success and env.robot.true_success
    env.close()


# ---------------------------------------------------------------- runner, labels, approval


class Script:
    """Answers for a terminal prompt, in order; records what was said."""

    def __init__(self, *answers):
        self.answers, self.said = list(answers), []

    def ask(self, prompt):
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)


class Fixed:
    def __init__(self, *labels):
        self.labels = list(labels)

    def label(self, env, auto_success, events):
        return self.labels.pop(0)


def test_a_session_records_the_label_not_the_estimators_opinion(stub_task, tmp_path):
    from skillforge.real.runner import Label, run_session

    env, cell = cell_env(stub_task)
    labels = Fixed(Label(True), Label(False, "wrong_placement", "on the rim"), Label(True))
    report, trials = run_session(env, PickPlace(), stub_task, [0, 1, 2], labels, tmp_path, mode="hardware", say=lambda s: None)
    assert report.n_success == 2 and report.n_rollouts == 3 and report.failure_counts == {"wrong_placement": 1}
    assert [t.auto_success for t in trials] == [True, True, True]  # the estimator thought all three worked
    side = json.loads((tmp_path / "real.json").read_text())
    assert side["mode"] == "hardware" and side["limits"]["ee_speed"] == 0.1
    assert side["auto_vs_label"]["agree"] == 2 and side["auto_vs_label"]["auto_only"] == 1
    t = side["trials"][1]
    assert t["label_failure"] == "wrong_placement" and t["note"] == "on the rim" and t["vlm_success"] is None
    assert t["targets"]["cube"]["pos"] == list(draw_placement(stub_task, 1)[1]["cube"].pos)
    assert t["measured"]["cube.place_error"] < 1e-9
    eps = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [e["success"] for e in eps] == [True, False, True] and "on the rim" in eps[1]["failure_detail"]
    with np.load(tmp_path / eps[0]["arrays"]) as z:
        T = eps[0]["n_steps"]
        assert z["observation.state"].shape[0] == T and z["action"].shape == (T, 8) and z["label_valid"].all()
        assert z["joint.command"].shape == z["joint.measured"].shape == (T, 3)
        assert np.allclose(z["joint.command"], z["joint.measured"])  # the fake tracks perfectly


def test_a_crashing_skill_is_recorded_and_the_arm_is_stopped(stub_task, tmp_path):
    from skillforge.real.runner import Label, run_trial

    env, cell = cell_env(stub_task)
    ep, trial = run_trial(env, Crashing(), stub_task, 0, Fixed(Label(False)), tmp_path)
    assert ep.failure == "exception" and "ZeroDivisionError" in ep.failure_detail
    assert trial.stopped == "exception:ZeroDivisionError" and cell.stopped and ep.n_steps == 11


def test_terminal_labeller_insists_on_a_valid_answer():
    from skillforge.real.runner import TerminalLabeller

    s = Script("maybe", "n", "broke", "grasp_slip", "slid out")
    lab = TerminalLabeller(s.ask, s.say).label(None, True, [])
    assert (lab.success, lab.failure, lab.note, lab.source) == (False, "grasp_slip", "slid out", "operator")
    assert "answer y or n" in s.said and "not a tag" in s.said
    s = Script("y", "")
    assert TerminalLabeller(s.ask, s.say).label(None, False, []).success


def test_terminal_operator_guides_the_placement_until_it_is_within_tolerance(stub_task):
    from skillforge.perception import Estimate
    from skillforge.real.runner import PlacementAborted, TerminalOperator

    _, targets = draw_placement(stub_task, 3)
    target = np.array(targets["cube"].pos)
    seen = [None, target + [0.03, -0.02, 0.0], target + [0.004, 0.0, 0.0]]

    def measure():
        p = seen.pop(0)
        return {} if p is None else {"cube": Estimate(pos=p, quat=np.array([1.0, 0, 0, 0]), found=True, confidence=1.0)}

    s = Script("", "", "")
    TerminalOperator(0.01, s.ask, s.say).place(3, targets, measure)
    assert s.said[1] == "the cameras cannot see it"
    assert s.said[2] == "36 mm off: move it -30 mm in x, +20 mm in y"
    assert s.said[3] == "ok, 4 mm off" and seen == []
    with pytest.raises(PlacementAborted):
        TerminalOperator(0.01, Script("a").ask, lambda t: None).place(3, targets, measure)
    TerminalOperator(0.01, Script("f").ask, lambda t: None).place(3, targets, measure)  # accepted unmeasured


def test_an_aborted_placement_skips_the_trial(stub_task, tmp_path):
    from skillforge.real.runner import Label, PlacementAborted, run_session

    class Fussy(FakeCell):
        def place(self, seed, targets, measure):
            if seed == 1:
                raise PlacementAborted("no cube")
            super().place(seed, targets, measure)

    cell = Fussy(stub_task)
    env = RealEnv(stub_task, cell, estimator=cell.estimator(), controller=PointController(DT), operator=cell)
    report, trials = run_session(env, PickPlace(), stub_task, [0, 1, 2], Fixed(Label(True), Label(True)), tmp_path, say=lambda s: None)
    assert report.n_rollouts == 2 and [t.seed for t in trials] == [0, 2]
    assert json.loads((tmp_path / "real.json").read_text())["aborted_seeds"] == [1]


def test_only_an_accepted_version_approved_by_name_may_run_on_hardware(stub_task, tmp_path):
    from skillforge.agents.library import SkillLibrary
    from skillforge.eval.regression import source_of
    from skillforge.real.runner import Approvals, NotApproved

    lib = SkillLibrary(tmp_path / "library")
    good = source_of(PickPlace)
    rejected = good.model_copy(update={"version": "9.9.9"})
    report, _ = evaluate(make_fake_cell_env, PickPlace(), stub_task, range(2))
    lib.add(good, report, accepted=True)
    lib.add(rejected, report, accepted=False)
    ap = Approvals(tmp_path / "approved.json")
    with pytest.raises(NotApproved, match="not approved"):
        ap.require_approved(stub_task.id, good)
    with pytest.raises(NotApproved, match="not accepted"):
        ap.approve(lib, stub_task.id, "9.9.9", by="Sam")
    with pytest.raises(NotApproved, match="not in the library"):
        ap.approve(lib, stub_task.id, "1.2.3", by="Sam")
    with pytest.raises(ValueError):
        ap.approve(lib, stub_task.id, good.version, by=" ")
    ap.approve(lib, stub_task.id, good.version, by="Sam")
    assert ap.require_approved(stub_task.id, good)["by"] == "Sam"
    tampered = good.model_copy(update={"code": good.code + "\n# edited after approval\n"})
    with pytest.raises(NotApproved, match="differs"):
        ap.require_approved(stub_task.id, tampered)


def test_cli_refuses_hardware_and_unapproved_skills(stub_task, tmp_path):
    from skillforge.agents.library import SkillLibrary
    from skillforge.eval.regression import source_of
    from skillforge.real import cli
    from skillforge.real.runner import NotApproved

    lib = SkillLibrary(tmp_path / "library")
    src = source_of(PickPlace)
    report, _ = evaluate(make_fake_cell_env, PickPlace(), stub_task, range(2))
    lib.add(src, report, accepted=True)
    common = ["--library", str(lib.root), "--approvals", str(tmp_path / "approved.json")]
    with pytest.raises(NotApproved):
        cli.main(["run", stub_task.id, "--skill", src.version, *common])
    cli.main(["approve", stub_task.id, src.version, "--by", "Sam", *common])
    with pytest.raises(SystemExit, match="no hardware layer"):
        cli.main(["run", stub_task.id, "--skill", src.version, *common])
    with pytest.raises(SystemExit, match="dry runs only"):
        cli.main(["run", stub_task.id, "--skill-file", "x.py", *common])
    assert list(cli._seeds("0-19")) == list(range(20))


def test_dry_run_session_from_the_cli_labels_with_true_sim_state(cube_task, tmp_path):
    from skillforge.real import cli

    out = tmp_path / "session"
    cli.main(["run", "franka_pick_cube_to_bin_hard", "--skill-file", "src/skillforge/skills/pick_place.py",
              "--seeds", "0-1", "--dry-run", "--speed", "0.25", "--out", str(out)])
    side = json.loads((out / "real.json").read_text())
    assert side["mode"] == "dry_run" and [t["label_source"] for t in side["trials"]] == ["sim_truth"] * 2
    assert side["limits"]["ee_speed"] == 0.25 and len(side["trials"]) == 2
    assert json.loads((out / "report.json").read_text())["n_rollouts"] == 2


# ---------------------------------------------------------------- system ID


def fake_log(latency, gain, T=120, n=4, seed=0):
    """A FakeRobotIO following a wandering command."""
    rng = np.random.default_rng(seed)
    robot = FakeRobotIO(np.zeros(n), latency_steps=latency, gain=gain)
    cmd = np.cumsum(rng.normal(0.0, 0.01, (T, n)), axis=0)
    meas = []
    for c in cmd:
        robot.send_joint_targets(c, 0.0)
        meas.append(robot.read_state().q)
    return np.zeros(n), cmd, np.array(meas)


@pytest.mark.parametrize(("latency", "gain"), [(0, 1.0), (2, 0.6), (5, 0.3)])
def test_sysid_recovers_a_known_latency_and_gain(latency, gain):
    from skillforge.real.sysid import fit_tracking

    fit = fit_tracking(*fake_log(latency, gain))
    assert fit.latency_steps == latency and abs(fit.gain - gain) < 1e-6 and fit.residual < 1e-9


def test_sysid_tolerates_encoder_noise_and_ignores_a_still_arm():
    from skillforge.real.sysid import fit_tracking

    start, cmd, meas = fake_log(3, 0.5, T=300)
    noisy = meas + np.random.default_rng(1).normal(0.0, 2e-4, meas.shape)
    fit = fit_tracking(start, cmd, noisy)
    assert fit.latency_steps == 3 and abs(fit.gain - 0.5) < 0.05
    assert fit_tracking(start, np.zeros_like(cmd), np.zeros_like(meas)) is None


def test_sysid_gripper_close_time_and_held_width():
    from skillforge.real.sysid import fit_gripper

    cmd = np.r_[np.zeros(10), np.ones(30), np.zeros(5)]
    width = np.r_[np.full(11, 0.08), np.linspace(0.08, 0.04, 9)[1:], np.full(21, 0.04), np.full(5, 0.08)]
    fit = fit_gripper(cmd, width, dt=0.05)
    assert abs(fit.close_time_s - 8 * 0.05) < 1e-9 and abs(fit.closed_width - 0.04) < 1e-9
    assert fit_gripper(np.zeros(20), np.full(20, 0.08), 0.05) is None  # never closed


def test_sysid_reads_a_session_and_proposes_a_randomization(stub_task, tmp_path):
    from skillforge.real.runner import Label, run_session
    from skillforge.real.sysid import equivalent_gain_scale, fit_session, propose_randomization

    cell = FakeCell(stub_task, latency_steps=2, gain=0.7)
    env = RealEnv(stub_task, cell, estimator=cell.estimator(), controller=PointController(DT), operator=cell)
    run_session(env, PickPlace(), stub_task, range(3), Fixed(*[Label(True)] * 3), tmp_path, mode="hardware", say=lambda s: None)
    cal = fit_session(tmp_path)
    assert cal.mode == "hardware" and cal.latency_steps == [2, 2, 2] and np.allclose(cal.gain, 0.7, atol=1e-6)
    assert "latency steps: 2" in cal.summary()
    curve = [(0.5, 0.4), (1.0, 0.6), (1.5, 0.8)]
    assert abs(equivalent_gain_scale(0.7, curve) - 1.25) < 1e-9 and equivalent_gain_scale(0.1, curve) == 0.5
    rnd = propose_randomization(stub_task, cal, curve)
    assert rnd.action_latency_steps.lo == rnd.action_latency_steps.hi == 2.0
    assert abs(rnd.actuator_gain_scale.lo - 1.25) < 1e-6 and abs(rnd.actuator_gain_scale.hi - 1.25) < 1e-6
    assert rnd.object_xy_jitter == stub_task.scene.randomization.object_xy_jitter  # untouched
    flat = [(0.5, 0.40), (1.0, 0.405), (1.5, 0.403)]  # what the Franka twin gives: no information
    kept = propose_randomization(stub_task, cal, flat)
    assert kept.actuator_gain_scale == stub_task.scene.randomization.actuator_gain_scale
    assert kept.action_latency_steps.lo == 2.0
    assert stub_task.scene.randomization.action_latency_steps.hi == 2  # the task itself is not edited


def test_sysid_finds_the_twins_latency_in_a_dry_run(cube_task):
    from skillforge.real import make_twin_env
    from skillforge.real.sysid import fit_tracking

    env = make_twin_env(cube_task)
    for seed in (0, 1, 3):
        skill = PickPlace()
        obs = env.reset(seed)
        skill.reset(cube_task)
        start, cmd, meas = obs.qpos.copy(), [], []
        for _ in range(150):
            obs, _ = env.step(skill.act(obs))
            cmd.append(env.command[0])
            meas.append(obs.qpos.copy())
        fit = fit_tracking(start, np.array(cmd), np.array(meas))
        assert fit.latency_steps == int(env.randomization_sample()["action_latency_steps"])
    env.close()


# ---------------------------------------------------------------- scoreboard and closing the loop


def session(stub_task, path, labels, *, mode="hardware", place_error=(0.0, 0.0), seeds=None):
    from skillforge.real.runner import run_session

    cell = FakeCell(stub_task, place_error=place_error)
    env = RealEnv(stub_task, cell, estimator=cell.estimator(), controller=PointController(DT), operator=cell)
    return run_session(env, PickPlace(), stub_task, seeds or range(len(labels)), Fixed(*labels), path, mode=mode, say=lambda s: None)


def test_scoreboard_pools_sessions_and_sets_them_against_sim(stub_task, tmp_path):
    from skillforge.real import scoreboard
    from skillforge.real.runner import Label

    ok, bad = Label(True), Label(False, "grasp_slip", "<b>slid</b>")
    session(stub_task, tmp_path / "a", [ok, ok, bad])
    session(stub_task, tmp_path / "b", [ok, bad], seeds=[3, 4])
    session(stub_task, tmp_path / "dry", [ok, ok], mode="dry_run")
    sim, _ = evaluate(make_fake_cell_env, PickPlace(), stub_task, range(30))
    board = scoreboard.build([tmp_path / "a", tmp_path / "b", tmp_path / "dry"], [sim])
    dry, real = board.rows  # sorted by mode: dry_run, hardware
    assert (real.mode, real.n, real.n_success, real.failure_counts) == ("hardware", 5, 3, {"grasp_slip": 2})
    assert len(real.sessions) == 2 and (real.sim_n, real.sim_n_success) == (30, 30)
    assert real.intervals_overlap is False  # 3/5 tops out at 88%, sim 30/30 starts at 89%
    assert real.estimator_vs_label.false_success == 2 and real.vlm_vs_label is None
    assert (dry.mode, dry.n, dry.n_success) == ("dry_run", 2, 2) and dry.intervals_overlap
    assert "3/5" in real.line() and "sim 30/30" in real.line()
    back = scoreboard.Scoreboard.model_validate_json(scoreboard.write_json(board, tmp_path / "board.json").read_text())
    assert back == board
    page = scoreboard.write_html(board, tmp_path / "board.html").read_text()
    assert "dry run (simulated twin)" in page and "real arm" in page and "intervals do not overlap" in page
    assert "<b>slid</b>" not in page


def test_a_real_failure_becomes_a_pinned_sim_task_and_a_baseline(stub_task, tmp_path):
    from skillforge.eval.regression import load_baselines, source_of
    from skillforge.real.failures import close_loop, pin_failure, write_task
    from skillforge.real.runner import Label
    from skillforge.sim import StubEnv

    session(stub_task, tmp_path / "s", [Label(True), Label(False, "grasp_miss")], place_error=(0.006, -0.004))
    side = json.loads((tmp_path / "s" / "real.json").read_text())
    trial = side["trials"][1]
    pinned = pin_failure(stub_task, trial)
    assert pinned.id == "stub_pick_cube_to_bin_real_seed1"
    rnd = pinned.scene.randomization
    assert rnd.object_xy_jitter.hi == 0.0 and rnd.obs_noise_std == stub_task.scene.randomization.obs_noise_std
    # the cube sits where it was measured: the seed's draw plus the operator's error
    _, targets = draw_placement(stub_task, 1)
    assert np.allclose(pinned.scene.object("cube").pose.pos[:2], np.array(targets["cube"].pos[:2]) + [0.006, -0.004])
    env = StubEnv(pinned)
    assert np.allclose(env.reset(123).objects["cube"].pos[:2], pinned.scene.object("cube").pose.pos[:2], atol=0.02)

    cases = close_loop(tmp_path / "s", stub_task, source_of(PickPlace), StubEnv, tmp_path / "tasks", tmp_path / "baselines", seeds=range(5))
    assert [c.seed for c in cases] == [1] and cases[0].label_failure == "grasp_miss"
    assert TaskSpec.model_validate(json.loads(cases[0].task_path.read_text())) == pinned
    assert load_baselines(tmp_path / "baselines")[0].task_id == pinned.id
    assert cases[0].baseline.n_success == 5 and not cases[0].reproduced  # the stub sim does not reproduce it
    with pytest.raises(FileExistsError):
        write_task(pinned, tmp_path / "tasks")
    assert stub_task.scene.object("cube").pose.pos == (0.5, -0.1, 0.02)  # the original task is untouched


def test_cli_sysid_and_scoreboard(stub_task, tmp_path, capsys):
    from skillforge.real import cli
    from skillforge.real.runner import Label

    session(stub_task, tmp_path / "s", [Label(True), Label(True)])
    cli.main(["sysid", str(tmp_path / "s")])
    assert json.loads((tmp_path / "s" / "calibration.json").read_text())["latency_steps"] == [0, 0]
    cli.main(["scoreboard", "--sessions", str(tmp_path / "s"), "--out", str(tmp_path / "board")])
    assert (tmp_path / "board" / "scoreboard.html").exists()
    assert "2/2" in capsys.readouterr().out


def test_a_judge_gives_a_third_opinion_that_is_recorded_but_never_the_result(stub_task, tmp_path):
    from types import SimpleNamespace

    from skillforge.real import scoreboard
    from skillforge.real.runner import Label, run_session

    class Camera:
        def capture(self, camera):
            return SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8))

    class SaysNo:
        calls = 0

        def judge(self, task, images):
            SaysNo.calls += 1
            return SimpleNamespace(success=False)

    cell = FakeCell(stub_task)
    env = RealEnv(stub_task, cell, Camera(), cell.estimator(), controller=PointController(DT), operator=cell)
    report, trials = run_session(env, PickPlace(), stub_task, [0, 1], Fixed(Label(True), Label(True)), tmp_path,
                                 mode="hardware", judge=SaysNo(), say=lambda s: None)
    assert report.n_success == 2 and [t.vlm_success for t in trials] == [False, False] and SaysNo.calls == 2
    row = scoreboard.build([tmp_path]).rows[0]
    assert row.vlm_vs_label.missed_success == 2 and row.vlm_vs_label.agree == 0


# ---------------------------------------------------------------- the operator's overlay


def looking_down(height=1.0, size=(120, 160), f=200.0):
    """A camera `height` above the world origin looking straight down: +u is world +x, +v is world -y."""
    from skillforge.perception import CameraFrame

    h, w = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    T[:3, 3] = [0.0, 0.0, height]
    return CameraFrame(rgb=np.zeros((h, w, 3), dtype=np.uint8), depth=np.ones((h, w), dtype=np.float32), K=K, T_world_cam=T)


def test_footprint_of_a_box_and_a_cylinder(stub_task):
    from skillforge.contracts import ObjectSpec
    from skillforge.real.overlay import footprint

    box = footprint(stub_task.scene.object("cube"), (0.5, -0.1, 0.02), np.pi / 2)
    assert box.shape == (5, 3) and np.allclose(box[0], box[-1])
    assert np.allclose(np.abs(box[:4, :2] - [0.5, -0.1]), 0.02) and np.allclose(box[:, 2], 0.02)
    can = ObjectSpec(name="can", shape="cylinder", size=(0.03, 0.05, 0.0))
    ring = footprint(can, (0.1, 0.2, 0.05), 0.0)
    assert np.allclose(np.linalg.norm(ring[:, :2] - [0.1, 0.2], axis=1), 0.03)


def test_overlay_draws_the_target_in_green_and_the_estimate_in_red(stub_task):
    from skillforge.perception import Estimate
    from skillforge.real import Placement
    from skillforge.real.overlay import GREEN, RED, placement_overlay

    spec = stub_task.scene.object("cube")
    frame = looking_down()
    target = Placement(pos=(0.1, 0.0, 0.02), quat=(1.0, 0.0, 0.0, 0.0), yaw_offset=0.0)
    img = placement_overlay(frame, spec, target)
    green = np.argwhere((img == GREEN).all(axis=2))
    assert len(green) > 20 and not (img == RED).all(axis=2).any() and not frame.rgb.any()  # the frame is not edited
    # the square's centre: 0.1 m in +x on the cube's top face, 0.96 m below the camera: 200 * 0.1 / 0.96 = 20.8 px right
    assert abs(green[:, 1].mean() - (80 + 20.8)) < 1.0 and abs(green[:, 0].mean() - 60) < 1.0
    est = Estimate(pos=np.array([-0.1, 0.05, 0.02]), quat=np.array([1.0, 0, 0, 0]), found=True, confidence=1.0)
    both = placement_overlay(frame, spec, target, est)
    red = np.argwhere((both == RED).all(axis=2))
    assert abs(red[:, 1].mean() - (80 - 20.8)) < 1.5 and abs(red[:, 0].mean() - (60 - 10.4)) < 1.5
    hidden = Estimate(pos=est.pos, quat=est.quat, found=False, confidence=0.0)
    assert not (placement_overlay(frame, spec, target, hidden) == RED).all(axis=2).any()
