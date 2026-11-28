"""T7: the lab RealEnv assembled from its real parts, with stand-ins only for the hardware:
the bridge server in --fake mode (the arm), the sim rendered from the lab camera's pose (the
camera), and an operator that places the cans in that sim. The estimator, controller, safety
and RealEnv are the real ones."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from skillforge.contracts import Action, TaskSpec  # noqa: E402
from skillforge.perception import SimFrameSource  # noqa: E402
from skillforge.real import SIM_HOME, FrankaBridgeRobotIO  # noqa: E402
from skillforge.real.lab import CAMERA, identical_cylinders, make_lab_env  # noqa: E402
from skillforge.sim import make_env  # noqa: E402

ROOT = Path(__file__).parent.parent
TASK = ROOT / "tasks" / "franka_stack_can_pyramid_lab.json"
sys.path.insert(0, str(Path(__file__).parent))
from test_cans import lab_camera, set_pose  # noqa: E402


@pytest.fixture
def server():
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "frankapy_bridge_server.py"), "--fake", "--port", "0"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    port = int(p.stdout.readline().split(":")[1].split()[0])
    yield port
    p.terminate()
    p.wait(5)


class SimCamera:
    """FrameSource that renders a scene MujocoEnv from the lab camera's pose, under the name "fixed"."""

    def __init__(self, env):
        self.src = SimFrameSource(env, virtual=(lab_camera(),), size=(480, 640), depth_noise=0.002)

    def capture(self, camera):
        assert camera == CAMERA
        f = self.src.capture("lab")
        f.masks = {}  # a real camera has no masks
        return f


class SimPlacer:
    """Operator: puts each can at its drawn target in the scene sim."""

    def __init__(self, env):
        self.env = env
        self.seen = []

    def place(self, seed, targets, measure):
        for name, t in targets.items():
            set_pose(self.env, name, np.array(t.pos))
        est = measure()
        self.seen.append({n: e.pos.copy() for n, e in est.items() if e.found})


def test_the_lab_task_is_wired():
    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    assert identical_cylinders(task)


def test_lab_env_resets_and_steps_with_names_on_the_placed_cans(server, tmp_path):
    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    scene = make_env(task)
    scene.reset(0)
    if scene.render() is None:
        scene.close()
        pytest.skip("no GL context")
    cam, placer = SimCamera(scene), SimPlacer(scene)
    env = make_lab_env(task, extrinsics="unused", background=False, robot=FrankaBridgeRobotIO(port=server), frames=cam, operator=placer,
                       picture=None)
    try:
        obs = env.reset(4)
        np.testing.assert_allclose(obs.qpos, SIM_HOME, atol=1e-9)
        targets = env._arm.targets
        assert set(obs.objects) == {o.name for o in task.scene.objects}
        for name, t in targets.items():  # each name is the can placed at its own target
            assert np.linalg.norm(obs.objects[name].pos[:2] - np.asarray(t.pos)[:2]) < 0.006, name
        start = obs.ee_pos.copy()
        for _ in range(10):
            obs, info = env.step(Action(mode="ee", ee_pos=start - [0, 0, 0.02], ee_quat=obs.ee_quat, gripper=0.0))
            assert not info.terminated, info.events
        assert obs.ee_pos[2] < start[2] - 0.005
    finally:
        env.close()
        cam.src.close()
        scene.close()


def test_a_dry_run_of_the_lab_pyramid_through_the_cli(tmp_path):
    """The command a hardware session uses, on the sim twin: a multi-object (AllOf) task through
    RealEnv and the session runner, arrays saved without T5's single-goal state vector."""
    from skillforge.real import cli

    out = tmp_path / "s"
    cli.main(["run", str(TASK), "--skill-file", "src/skillforge/skills/place_in_order.py:PlaceInOrder",
              "--dry-run", "--seeds", "0-0", "--out", str(out)])
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_success"] == 1
    z = np.load(out / "arrays" / "episode_0.npz")
    assert "joint.command" in z.files and "joint.measured" in z.files and "observation.state_missing" in z.files


def test_the_first_can_is_placed_on_its_spot_with_measure_once_perception():
    """The lab pipeline on the twin: lab camera pose, can estimator measured once, bookkeeping after.
    (Live estimates let the skill drop the can mid-lift on the arm, 2026-11-27, trial 5.)"""
    from skillforge.real import MujocoController, MujocoTwin, RealEnv, SafetyLimits, Workspace
    from skillforge.real.background import MeasureOnce
    from skillforge.real.cans import IdenticalCylinderEstimator
    from skillforge.skills import PlaceInOrder

    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    if make_env(task).render() is None:
        pytest.skip("no GL context")
    limits = SafetyLimits(ee_speed=0.1)
    twin = MujocoTwin(task)
    ctl = MujocoController(task, max_ee_speed=0.1, max_ee_ang_speed=limits.ee_ang_speed)
    cam = SimCamera(twin.env)
    env = RealEnv(task, twin, None, MeasureOnce(IdenticalCylinderEstimator(), cam, CAMERA), controller=ctl,
                  operator=twin, limits=limits, workspace=Workspace.from_task(task, include=[ctl.fk(ctl.home)[0]]),
                  cameras=(CAMERA,), report_contact_force=False)
    skill = PlaceInOrder()
    try:
        obs = env.reset(0)
        skill.reset(task)
        events = []
        for _ in range(420):
            obs, info = env.step(skill.act(obs))
            events += info.events
            if "release:cyl_1" in events:
                break
        m, d = twin.env.model, twin.env.data
        spot = np.array(task.scene.region("spot_cyl_1").center)
        assert "grasp:cyl_1" in events and "release:cyl_1" in events
        assert not any(e.startswith(("grasp_slip", "grasp_miss")) for e in events), events
        assert np.linalg.norm(d.xpos[m.body("cyl_1").id][:2] - spot[:2]) < 0.01
    finally:
        env.close()
        cam.src.close()


def test_a_can_dropped_mid_carry_is_found_and_placed():
    """The point of watching the table: the can slips mid-carry and lands elsewhere; the perception
    notices (fingers collapse), finds it on the next clear view, and the skill re-grasps it there."""
    import mujoco

    from skillforge.real import MujocoController, MujocoTwin, RealEnv, SafetyLimits, Workspace
    from skillforge.real.background import TablePerception
    from skillforge.real.cans import IdenticalCylinderEstimator
    from skillforge.skills import PlaceInOrder

    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    if make_env(task).render() is None:
        pytest.skip("no GL context")
    limits = SafetyLimits(ee_speed=0.1)
    twin = MujocoTwin(task)
    ctl = MujocoController(task, max_ee_speed=0.1, max_ee_ang_speed=limits.ee_ang_speed)
    cam = SimCamera(twin.env)
    k = [0]
    per = TablePerception(IdenticalCylinderEstimator(), cam, CAMERA, sync=True, clock=lambda: k[0] * 0.05)
    env = RealEnv(task, twin, None, per, controller=ctl, operator=twin, limits=limits,
                  workspace=Workspace.from_task(task, include=[ctl.fk(ctl.home)[0]]), cameras=(CAMERA,),
                  report_contact_force=False)
    skill = PlaceInOrder()
    events, dropped = [], False
    try:
        obs = env.reset(0)
        skill.reset(task)
        for k[0] in range(700):
            obs, info = env.step(skill.act(obs))
            events += info.events + per.events
            per.events.clear()
            carrying = "grasp:cyl_1" in events and "release:cyl_1" not in events
            if not dropped and carrying and obs.ee_pos[2] > 0.15:
                m, d = twin.env.model, twin.env.data
                j = m.joint("cyl_1_free")
                d.qpos[j.qposadr[0]:j.qposadr[0] + 7] = [0.50, -0.29, 0.0419, 1, 0, 0, 0]  # a free spot
                d.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0
                mujoco.mj_forward(m, d)
                dropped = True
            if "release:cyl_1" in events:
                break
        m, d = twin.env.model, twin.env.data
        spot = np.array(task.scene.region("spot_cyl_1").center)
        assert dropped and "perception:slip:cyl_1" in events and "perception:moved:cyl_1" in events
        assert events.count("grasp:cyl_1") == 2  # grasped again, where it landed
        assert np.linalg.norm(d.xpos[m.body("cyl_1").id][:2] - spot[:2]) < 0.01
    finally:
        env.close()
        cam.src.close()


class _SaggingTwin:
    """The twin, but the arm settles `sag` (rad, per joint) short of every command (only while the
    gripper is closed with `loaded=True`: on the arm the wrist lagged while carrying a can)."""

    def __init__(self, twin, sag, loaded=False):
        self.twin, self.sag, self.loaded = twin, np.asarray(sag, float), loaded

    def send_joint_targets(self, q, g):
        on = not self.loaded or g > 0.5
        self.twin.send_joint_targets(np.asarray(q, float) - (self.sag if on else 0.0), g)

    def __getattr__(self, n):
        return getattr(self.twin, n)


WRIST_LAG = [0, 0, 0, 0, 0, 0.068, 0]  # lab, 2026-11-28 (11:23, 11:25, 12:16): joint 6 held 0.05-0.07 rad short


@pytest.mark.parametrize("sag, settle_gain, bias_max, placed", [
    ([0, 0.005, 0, 0, 0, 0, 0], 0.0, None, False),
    ([0, 0.005, 0, 0, 0, 0, 0], 2.0, None, True),
    (WRIST_LAG, 2.0, 0.03, False),  # the old cap: hovers ~1 cm over the spot, as on the arm
    (WRIST_LAG, 2.0, None, True),  # the current cap
])
def test_an_arm_that_settles_short_still_places_the_can(sag, settle_gain, bias_max, placed):
    """Lab, 2026-11-28, trial 6: the Franka settled ~2 mm short of its command holding a can over its
    spot; the skill waits for 1.5 mm alignment before setting it down, and waited forever. RealEnv's
    steady-state correction closes the gap (off: stuck, as on the arm; on: placed). Later trials: the
    wrist (joint 6) held 0.05-0.07 rad short while setting the first can down; a 0.03 rad cap on the
    correction could not close that, the skill timed out and held the can ~1 cm over its spot."""
    from skillforge.real import MujocoController, MujocoTwin, RealEnv, SafetyLimits, Workspace
    from skillforge.real.background import TablePerception
    from skillforge.real.cans import IdenticalCylinderEstimator
    from skillforge.skills import PlaceInOrder

    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    if make_env(task).render() is None:
        pytest.skip("no GL context")
    limits = SafetyLimits(ee_speed=0.1)
    twin = MujocoTwin(task)
    ctl = MujocoController(task, max_ee_speed=0.1, max_ee_ang_speed=limits.ee_ang_speed)
    cam = SimCamera(twin.env)
    k = [0]
    per = TablePerception(IdenticalCylinderEstimator(), cam, CAMERA, sync=True, clock=lambda: k[0] * 0.05)
    env = RealEnv(task, _SaggingTwin(twin, sag, loaded=sag is WRIST_LAG), None, per, controller=ctl, operator=twin,
                  limits=limits, workspace=Workspace.from_task(task, include=[ctl.fk(ctl.home)[0]]),
                  cameras=(CAMERA,), report_contact_force=False, settle_gain=settle_gain)
    if bias_max is not None:
        env._arm.bias_max = bias_max
    skill = PlaceInOrder()
    events = []
    try:
        obs = env.reset(0)
        skill.reset(task)
        for k[0] in range(600):
            obs, info = env.step(skill.act(obs))
            events += info.events
            if "release:cyl_1" in events:
                break
        assert "grasp:cyl_1" in events and ("release:cyl_1" in events) is placed
        if placed:
            m, d = twin.env.model, twin.env.data
            spot = np.array(task.scene.region("spot_cyl_1").center)
            assert np.linalg.norm(d.xpos[m.body("cyl_1").id][:2] - spot[:2]) < 0.01
    finally:
        env.close()
        cam.src.close()


class _AllOnSpots:
    """Estimator stand-in: every can sits on its pyramid spot; `held` names the can in the hand."""

    needs_frames = False

    def __init__(self, task):
        self.task, self.held = task, None

    def reset(self, task):
        pass

    def estimate(self, frames, obs=None):
        from skillforge.perception import Estimate

        return {o.name: Estimate(pos=np.array(self.task.scene.region(f"spot_{o.name}").center, float),
                                 quat=np.array([1.0, 0, 0, 0]), found=True, confidence=1.0)
                for o in self.task.scene.objects}


def test_success_waits_until_the_last_can_is_released():
    """Lab, 2026-11-28 (11:10 and 11:47 trials): bookkeeping carries a held can with the hand, so the
    last can was 'on its spot' while still gripped; success ended the trial and left it in the gripper."""
    from skillforge.real import MujocoController, MujocoTwin, RealEnv, SafetyLimits, Workspace

    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    limits = SafetyLimits(ee_speed=0.1)
    twin = MujocoTwin(task)
    ctl = MujocoController(task, max_ee_speed=0.1, max_ee_ang_speed=limits.ee_ang_speed)
    est = _AllOnSpots(task)
    env = RealEnv(task, twin, None, est, controller=ctl, operator=twin, limits=limits,
                  workspace=Workspace.from_task(task, include=[ctl.fk(ctl.home)[0]]), cameras=(CAMERA,),
                  report_contact_force=False)
    try:
        obs = env.reset(0)
        hold = Action(mode="ee", ee_pos=obs.ee_pos, ee_quat=obs.ee_quat, gripper=0.0)
        est.held = "cyl_6"
        assert not env.step(hold)[1].success
        est.held = None
        assert env.step(hold)[1].success
    finally:
        env.close()
