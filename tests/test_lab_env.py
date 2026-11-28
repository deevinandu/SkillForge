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
