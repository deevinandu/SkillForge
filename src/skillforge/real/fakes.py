"""T7: stand-ins for the arm, the table and the operator. No hardware, no ROS.

- `FakeRobotIO`: joints that follow their targets with a set latency and gain.
- `FakeCell`: `FakeRobotIO` for the stub point robot plus a table of objects that follow the
  stub env's rules (grip within 2 cm to pick up, carried while closed). It is also its own
  operator and its own estimator, so `RealEnv` runs end to end with no MuJoCo.
- `MujocoTwin`: a `MujocoEnv` behind the `RobotIO` interface. This is the dry run: the commands
  `RealEnv` would send to the arm go to the simulated arm instead.

`make_fake_cell_env` and `make_twin_env` are module-level, so they work with `evaluate_parallel`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from skillforge.contracts import Action, Observation, TaskSpec
from skillforge.perception import Estimate

from .controller import MujocoController, PointController
from .core import Measure, Placement, RealEnv, RobotState
from .safety import SafetyLimits, Workspace

GRASP_RADIUS = 0.02  # metres, as in `sim.stub_env`


class FakeRobotIO:
    """Each `send_joint_targets` is one control period. The target sent `latency_steps` ago is
    applied, and the joints close `gain` of the remaining distance (1 = perfect tracking)."""

    def __init__(self, home, *, dt: float = 0.05, latency_steps: int = 0, gain: float = 1.0, max_width: float = 0.08):
        self._home = np.asarray(home, dtype=float).copy()
        self.dt = dt
        self.latency_steps = latency_steps
        self.gain = gain
        self.max_width = max_width
        self.force: np.ndarray | None = None  # set by a test to simulate contact
        self.sent: list[tuple[np.ndarray, float]] = []  # every target received, stopped or not
        self.stopped = False
        self.home()

    def home(self) -> None:
        self.q = self._home.copy()
        self.qvel = np.zeros_like(self.q)
        self.grip = 0.0
        self.t = 0.0
        self.stopped = False
        self._queue: deque[tuple[np.ndarray, float]] = deque()
        self._events: list[str] = []

    def stop(self) -> None:
        self.stopped = True
        self._queue.clear()
        self.qvel = np.zeros_like(self.q)

    def send_joint_targets(self, q: np.ndarray, gripper: float) -> None:
        q = np.asarray(q, dtype=float).copy()
        self.sent.append((q, float(gripper)))
        self.t += self.dt
        if self.stopped:
            return
        self._queue.append((q, float(gripper)))
        if len(self._queue) > self.latency_steps:
            target, grip = self._queue.popleft()
            new = self.q + self.gain * (target - self.q)
            self.qvel = (new - self.q) / self.dt
            self.q = new
            self._gripper(grip)

    def _gripper(self, grip: float) -> None:
        self.grip = grip

    def read_state(self) -> RobotState:
        events, self._events = self._events, []
        return RobotState(
            t=self.t,
            q=self.q.copy(),
            qvel=self.qvel.copy(),
            gripper_width=self.max_width * (1.0 - self.grip),
            ee_force=None if self.force is None else np.asarray(self.force, dtype=float).copy(),
            events=events,
        )


class _Truth:
    """A PoseEstimator that reads a fake's own object poses: perfect perception."""

    needs_frames = False

    def __init__(self, poses):
        self._poses = poses

    def reset(self, task: TaskSpec) -> None:
        pass

    def estimate(self, frames, obs: Observation) -> dict[str, Estimate]:
        return {
            name: Estimate(pos=np.asarray(p, dtype=float).copy(), quat=np.asarray(q, dtype=float).copy(), found=True, confidence=1.0)
            for name, (p, q) in self._poses().items()
        }


class FakeCell(FakeRobotIO):
    """The stub robot and its table. `place_error` (metres, xy) is added to every placement, to
    model an operator who is a little off."""

    def __init__(self, task: TaskSpec, *, latency_steps: int = 0, gain: float = 1.0, place_error=(0.0, 0.0)):
        self.task = task
        self.place_error = np.array([*place_error, 0.0], dtype=float)
        self._obj: dict[str, np.ndarray] = {}
        self._quat: dict[str, np.ndarray] = {}
        self._rest: dict[str, float] = {}
        self._held: str | None = None
        home = np.array(task.scene.robot.base_pose.pos, dtype=float) + [0.4, 0.0, 0.3]
        super().__init__(home, dt=1.0 / task.control_hz, latency_steps=latency_steps, gain=gain,
                         max_width=task.scene.robot.max_gripper_width)

    def home(self) -> None:
        super().home()
        self._held = None

    # operator
    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        for name, t in targets.items():
            self._obj[name] = np.asarray(t.pos, dtype=float) + self.place_error
            self._quat[name] = np.asarray(t.quat, dtype=float)
            self._rest[name] = self._obj[name][2]

    # estimator
    def estimator(self) -> _Truth:
        return _Truth(lambda: {n: (p, self._quat[n]) for n, p in self._obj.items()})

    def _gripper(self, grip: float) -> None:
        was_closed, closed = self.grip > 0.5, grip > 0.5
        self.grip = grip
        if closed and not was_closed and self._held is None:
            near = [n for n, p in self._obj.items() if np.linalg.norm(p - self.q) < GRASP_RADIUS]
            self._held = near[0] if near else None
            self._events.append(f"grasp:{near[0]}" if near else "grasp_miss")
        if not closed and self._held is not None:
            self._events.append(f"release:{self._held}")
            self._held = None
        for n, p in self._obj.items():
            if n == self._held:
                p[:] = self.q
            else:
                p[2] = self._rest[n]  # an object that is let go drops back onto the table


class MujocoTwin:
    """A simulated arm and table behind `RobotIO`: the dry run.

    It is its own operator (`place` resets the sim with the seed, which puts the objects exactly
    on target and applies the seed's physical randomization) and offers perfect perception
    (`estimator()`). For perception in the loop pass T6's `SimFrameSource(twin.env, ...)` and
    `DepthEstimator` to `RealEnv` instead.
    """

    def __init__(self, task: TaskSpec, **env_kwargs):
        from skillforge.sim.mujoco_env import MujocoEnv  # lazy: mujoco is an optional dependency

        self.task = task
        self.env = MujocoEnv(task, **env_kwargs)
        self._obs = self.env.reset(0)
        self._events: list[str] = []
        self.stopped = False
        self.true_success = False  # on TRUE sim state, for scoring the estimator's success call

    def home(self) -> None:
        self.stopped = False  # the arm itself goes home when `place` resets the sim

    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        self._obs = self.env.reset(seed)
        self._events = []
        self.true_success = False

    def stop(self) -> None:
        self.stopped = True

    def send_joint_targets(self, q: np.ndarray, gripper: float) -> None:
        if self.stopped:
            return  # the sim is frozen; a real arm would hold position
        self._obs, info = self.env.step(Action(mode="joint", joint_targets=np.asarray(q, dtype=float), gripper=gripper))
        self._events += info.events
        self.true_success = info.success

    def read_state(self) -> RobotState:
        o = self._obs
        events, self._events = self._events, []
        return RobotState(t=o.t, q=o.qpos.copy(), qvel=o.qvel.copy(), gripper_width=o.gripper_width,
                          ee_force=None if o.contact_force is None else o.contact_force.copy(), events=events)

    def randomization_sample(self) -> dict[str, float]:
        return self.env.randomization_sample()

    def render(self, camera: str | None = None):
        return self.env.render(camera)

    def estimator(self) -> _Truth:
        def poses():
            m, d = self.env.model, self.env.data
            return {o.name: (d.xpos[m.body(o.name).id], d.xquat[m.body(o.name).id])
                    for o in self.task.scene.objects if not o.fixed}

        return _Truth(poses)

    def close(self) -> None:
        self.env.close()


# Factories.

SIM_SPEED = SafetyLimits(ee_speed=0.5, ee_ang_speed=2.0, joint_speed=10.0, force=1e9)
"""The sim's own speeds and no force stop: for comparing a dry run with a plain sim run."""


def make_fake_cell_env(task: TaskSpec, *, limits: SafetyLimits | None = None, **cell_kwargs) -> RealEnv:
    limits = limits or SafetyLimits()
    cell = FakeCell(task, **cell_kwargs)
    controller = PointController(1.0 / task.control_hz, max_ee_speed=limits.ee_speed)
    return RealEnv(task, cell, estimator=cell.estimator(), controller=controller, operator=cell, limits=limits)


def make_twin_env(task: TaskSpec, *, limits: SafetyLimits | None = None) -> RealEnv:
    """A dry run with perfect perception at the real-robot speed limits."""
    limits = limits or SafetyLimits()
    twin = MujocoTwin(task)
    controller = MujocoController(task, max_ee_speed=limits.ee_speed, max_ee_ang_speed=limits.ee_ang_speed)
    workspace = Workspace.from_task(task, include=[controller.fk(controller.home)[0]])
    return RealEnv(task, twin, estimator=twin.estimator(), controller=controller, operator=twin,
                   limits=limits, workspace=workspace)


def make_depth_twin_env(task: TaskSpec, *, limits: SafetyLimits | None = None, depth_noise: float = 0.0,
                        segment: str = "mask") -> RealEnv:
    """A dry run that sees the table through T6's depth cameras instead of true poses.

    The estimator's success call can then be wrong, as it can on hardware; `twin.true_success`
    is the reference. Needs a GL backend.
    """
    from skillforge.perception import DepthEstimator, SimFrameSource, overview_camera

    limits = limits or SafetyLimits()
    twin = MujocoTwin(task)
    controller = MujocoController(task, max_ee_speed=limits.ee_speed, max_ee_ang_speed=limits.ee_ang_speed)
    workspace = Workspace.from_task(task, include=[controller.fk(controller.home)[0]])
    bx, by = task.scene.robot.base_pose.pos[:2]
    frames = SimFrameSource(twin.env, virtual=(overview_camera((bx, by)),), depth_noise=depth_noise)
    estimator = DepthEstimator(segment=segment, colors=frames.object_colors() if segment == "color" else None)
    return RealEnv(task, twin, frames, estimator, controller=controller, operator=twin, limits=limits,
                   workspace=workspace, cameras=("overview", "wrist"))


def make_sim_speed_twin_env(task: TaskSpec) -> RealEnv:
    return make_twin_env(task, limits=SIM_SPEED)
