"""T7: `RealEnv`, a `SimEnv` over a real arm. This file is ROS-free and hardware-free.

`RealEnv` talks to the arm through a `RobotIO` (joint state in, joint targets out) and to the
cameras through T6's `FrameSource`. The ROS 2 node (not written yet) implements both; the fakes
in `real/fakes.py` implement them for tests and dry runs.

What is the same as sim: the controller. `Action(mode="ee")` goes through a `Controller` that
runs the sim's own reference generator and IK on a kinematic model, so the controller is not a
source of sim-to-real gap.

What is different from sim, by necessity:
- object poses come from an estimator (T6), there is no true state;
- `StepInfo.success` is the estimator's opinion. It only ends the episode early. The recorded
  result on hardware is the operator's label, which the runner (not written yet) collects;
- `reset(seed)` cannot place objects. It draws the same placement the sim would draw for that
  seed and asks an `Operator` to put the object there;
- friction, mass, gain, noise and latency cannot be drawn. `randomization_sample()` reports them
  at nominal and sets `real.physical_nominal` to 1 to say so.

Nothing in this file has run on a real arm.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from skillforge.contracts import Action, Observation, StepInfo, TaskSpec
from skillforge.contracts.specs import success_objects
from skillforge.perception import Estimate, FrameSource, PerceivedEnv, PoseEstimator
from skillforge.perception.estimators import quat_yaw, wrap
from skillforge.sim import is_success

from .safety import Safety, SafetyLimits, Watchdog, Workspace


@dataclass
class RobotState:
    t: float  # seconds, the robot's own clock
    q: np.ndarray  # arm joint positions, radians
    qvel: np.ndarray
    gripper_width: float  # metres, measured finger opening
    ee_force: np.ndarray | None = None  # (3,) N at the end effector, if the arm estimates it
    events: list[str] = field(default_factory=list)  # contact events; only a simulated twin has them


@runtime_checkable
class RobotIO(Protocol):
    def read_state(self) -> RobotState: ...

    def send_joint_targets(self, q: np.ndarray, gripper: float) -> None:
        """Joint position targets for the next control period; gripper 0 = open, 1 = closed."""

    def stop(self) -> None:
        """Hold position and ignore targets until `home()`."""

    def home(self) -> None:
        """Move to the home posture and accept targets again. Blocks until there."""


@runtime_checkable
class Controller(Protocol):
    """Turns an `Action` into joint targets. Holds the reference pose between steps."""

    def reset(self, q: np.ndarray) -> None:
        """Start the reference at the measured joint positions."""

    def solve(self, action: Action) -> tuple[np.ndarray, list[str]]:
        """Joint targets for this control period and controller events (`ik_failure`)."""

    def sync(self, q: np.ndarray) -> None:
        """The command actually sent was `q`, not what `solve` returned: restart the reference there."""

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool-point position (3,) and quaternion (w, x, y, z) for joint positions `q`."""


@dataclass(frozen=True)
class Placement:
    """Where one object should start for a seed."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    yaw_offset: float  # radians about world z, relative to the task's nominal pose


Measure = Callable[[], dict[str, Estimate]]


class PlacementAborted(RuntimeError):
    pass


@runtime_checkable
class Operator(Protocol):
    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        """Put every object at its target and return once done. `measure()` gives the current
        estimates, for an overlay or a tolerance check. Raise `PlacementAborted` to give up."""


class AutoOperator:
    """Confirms at once. For sessions with no object on the table, and for fakes that place themselves."""

    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        pass


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def draw_placement(task: TaskSpec, seed: int) -> tuple[dict[str, float], dict[str, Placement]]:
    """The randomization sample and object placements `MujocoEnv.reset(seed)` draws.

    The draws are made in the same order from the same generator, so a seed means the same
    placement on the table as in sim (tested against `MujocoEnv`). Physical parameters are
    drawn too, to keep the generator in step, and returned for the record; hardware cannot
    apply them.
    """
    rng = np.random.default_rng(seed)
    rnd = task.scene.randomization
    u = lambda r: float(rng.uniform(r.lo, r.hi))
    s = {
        "object_xy_jitter": u(rnd.object_xy_jitter),
        "object_yaw_jitter": u(rnd.object_yaw_jitter),
        "friction_scale": u(rnd.friction_scale),
        "mass_scale": u(rnd.mass_scale),
        "actuator_gain_scale": u(rnd.actuator_gain_scale),
        "obs_noise_std": u(rnd.obs_noise_std),
        "action_latency_steps": float(round(u(rnd.action_latency_steps))),
    }
    targets = {}
    for o in task.scene.objects:
        if o.fixed:
            continue
        dx, dy = rng.uniform(-s["object_xy_jitter"], s["object_xy_jitter"], size=2)
        dyaw = float(rng.uniform(-s["object_yaw_jitter"], s["object_yaw_jitter"]))
        s[f"{o.name}.dx"], s[f"{o.name}.dy"], s[f"{o.name}.dyaw"] = float(dx), float(dy), dyaw
        pos = np.asarray(o.pose.pos, dtype=float) + [dx, dy, 0.0]
        qz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
        quat = _quat_mul(qz, np.asarray(o.pose.quat, dtype=float))
        targets[o.name] = Placement(tuple(map(float, pos)), tuple(map(float, quat)), dyaw)
    return s, targets


PHYSICAL_NOMINAL = {
    "friction_scale": 1.0,
    "mass_scale": 1.0,
    "actuator_gain_scale": 1.0,
    "obs_noise_std": 0.0,
    "action_latency_steps": 0.0,
}


class _Arm:
    """The robot half of `RealEnv`: a SimEnv whose observations have no objects. `RealEnv` wraps
    it in T6's `PerceivedEnv`, which fills the objects in from the cameras."""

    report_contact_force = True  # set by RealEnv; False hides the arm's force estimate from skills

    def __init__(self, task, robot, controller, safety, operator, measure, realtime, clock, sleep):
        self.task: TaskSpec = task
        self.robot: RobotIO = robot
        self.controller: Controller = controller
        self.safety: Safety = safety
        self.operator: Operator = operator
        self.measure: Measure = measure
        self.realtime = realtime
        self.clock = clock
        self.sleep = sleep
        self.dt = 1.0 / task.control_hz
        self.watchdog = Watchdog(safety.limits.watchdog_periods * self.dt, clock)
        self.stopped: str | None = None  # why the arm was stopped, or None
        self.sample: dict[str, float] = {}
        self.targets: dict[str, Placement] = {}
        self.safety_log: list[tuple[int, str]] = []  # (step, event) for the session record
        self._k = 0
        self._t = 0.0
        self._t0 = 0.0
        self._q_cmd = np.zeros(0)
        self._grip = 0.0

    # -- contract ----------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        self.robot.home()
        self.safety.reset()
        self.stopped = None
        self.safety_log = []
        self.sample, self.targets = draw_placement(self.task, seed)
        self.operator.place(seed, self.targets, self.measure)
        state = self.robot.read_state()
        self.controller.reset(state.q)
        self._q_cmd = np.asarray(state.q, dtype=float).copy()
        self._grip = 0.0
        home, _ = self.controller.fk(self._q_cmd)
        if not self.safety.workspace.contains(home):
            raise RuntimeError(f"home tool point {np.round(home, 3)} is outside the workspace {self.safety.workspace}")
        self._k, self._t, self._t0 = 0, 0.0, self.clock()
        self.watchdog.feed()
        return self.observe(state)

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        self._k += 1
        self._t += self.dt
        if self.stopped is not None:
            return self.observe(self.robot.read_state()), StepInfo(success=False, terminated=True)
        events: list[str] = []
        if self.realtime and self.watchdog.expired():
            # The arm has been holding its last target. Restart the reference from where it is.
            events.append("safety:watchdog")
            self._q_cmd = np.asarray(self.robot.read_state().q, dtype=float).copy()
            self.controller.sync(self._q_cmd)
        self.watchdog.feed()
        act, ev = self.safety.check_action(action, self.controller.fk)
        events += ev
        self._grip = float(action.gripper)
        if act is not None:  # None: refused, hold the last command
            q_des, ev = self.controller.solve(act)
            events += ev
            q, ev = self.safety.limit_joints(q_des, self._q_cmd, self.dt)
            events += ev
            if q is not q_des:
                self.controller.sync(q)
            self._q_cmd = q
        self.robot.send_joint_targets(self._q_cmd, self._grip)
        if self.realtime:
            wait = self._t0 + self._k * self.dt - self.clock()
            if wait > 0:
                self.sleep(wait)
        state = self.robot.read_state()
        events += state.events
        if self.safety.force_exceeded(state.ee_force):
            events.append("safety:force")
            self.stop("force")
        self.safety_log += [(self._k, e) for e in events if e.startswith("safety:")]
        return self.observe(state), StepInfo(success=False, terminated=self.stopped is not None, events=events)

    def randomization_sample(self) -> dict[str, float]:
        return dict(self.sample)

    def render(self, camera: str | None = None) -> np.ndarray | None:
        return None

    def close(self) -> None:
        if hasattr(self.robot, "close"):
            self.robot.close()

    # -- internals ---------------------------------------------------------

    def stop(self, reason: str) -> None:
        if self.stopped is None:
            self.stopped = reason
            self.robot.stop()

    def observe(self, state: RobotState) -> Observation:
        pos, quat = self.controller.fk(np.asarray(state.q, dtype=float))
        return Observation(
            t=self._t,
            qpos=np.asarray(state.q, dtype=float).copy(),
            qvel=np.asarray(state.qvel, dtype=float).copy(),
            ee_pos=np.asarray(pos, dtype=float).copy(),
            ee_quat=np.asarray(quat, dtype=float).copy(),
            gripper_width=float(state.gripper_width),
            contact_force=None if (state.ee_force is None or not self.report_contact_force)
            else np.asarray(state.ee_force, dtype=float).copy(),
        )


class RealEnv:
    """`contracts.SimEnv` over a `RobotIO`, camera frames and a pose estimator.

    frames, estimator: T6's `FrameSource` and `PoseEstimator`. With no estimator the
        observations carry no objects (an arm-only session) and success is never reported.
    controller: required. `MujocoController(task)` for a real arm.
    operator: places the objects at reset. Default: `AutoOperator`, which does nothing.
    workspace: default `Workspace.from_task(task)`.
    realtime: pace `step` to the control period with `sleep` and run the watchdog on `clock`.
        Off by default, so fakes and dry runs go as fast as they can.
    perception: extra keyword arguments for `PerceivedEnv` (e.g. `estimate_hz`).
    """

    def __init__(
        self,
        task: TaskSpec,
        robot: RobotIO,
        frames: FrameSource | None = None,
        estimator: PoseEstimator | None = None,
        *,
        controller: Controller,
        operator: Operator | None = None,
        limits: SafetyLimits | None = None,
        workspace: Workspace | None = None,
        cameras: tuple[str, ...] = ("fixed", "wrist"),
        realtime: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        perception: dict | None = None,
        report_contact_force: bool = True,
    ):
        if estimator is not None and estimator.needs_frames and frames is None:
            raise ValueError("this estimator needs camera frames: pass `frames`")
        self.task = task
        self.robot = robot
        self.frames = frames
        self.estimator = estimator
        self.cameras = tuple(cameras)
        self.safety = Safety(limits or SafetyLimits(), workspace or Workspace.from_task(task))
        self._arm = _Arm(task, robot, controller, self.safety, operator or AutoOperator(), self._measure,
                         realtime, clock, sleep)
        # False: skills get no contact force. A torque-based estimate (the Franka's O_F_ext_hat_K)
        # reads several newtons while the arm moves with nothing touching; a skill that stops
        # descending on |F| > 5 N then closes in mid-air (lab, 2026-11-27, first pyramid trial).
        # The safety force limit still reads the raw value either way.
        self._arm.report_contact_force = report_contact_force
        self._env = self._arm if estimator is None else PerceivedEnv(
            self._arm, estimator, source=frames, cameras=self.cameras, **(perception or {})
        )
        self._start: dict[str, np.ndarray] = {}
        self._measured: dict[str, float] = {}
        self._obs: Observation | None = None

    # -- contract ----------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        if self.estimator is not None:
            self.estimator.reset(self.task)  # `Operator.place` may call `measure()` before the wrapper resets it
        obs = self._env.reset(seed)
        self._start = {n: s.pos.copy() for n, s in obs.objects.items()}
        self._measured = {}
        for name, target in self._arm.targets.items():
            if name in obs.objects:
                nominal = np.asarray(self.task.scene.object(name).pose.pos, dtype=float)
                p = obs.objects[name].pos
                self._measured[f"{name}.placed_dx"] = float(p[0] - nominal[0])
                self._measured[f"{name}.placed_dy"] = float(p[1] - nominal[1])
                self._measured[f"{name}.place_error"] = float(np.linalg.norm(p[:2] - np.asarray(target.pos)[:2]))
                # yaw about world z relative to the nominal pose; meaningless for a round object
                nominal_yaw = quat_yaw(np.asarray(self.task.scene.object(name).pose.quat, dtype=float))
                self._measured[f"{name}.placed_dyaw"] = wrap(quat_yaw(obs.objects[name].quat) - nominal_yaw, 2 * np.pi)
        self._obs = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        try:
            obs, info = self._env.step(action)
        except Exception as e:  # robot IO, controller or estimator: stop first, then report
            name = type(e).__name__
            self.stop(f"exception:{name}")
            self._arm.safety_log.append((self._arm._k, f"safety:exception:{name}"))
            return self._obs, StepInfo(success=False, terminated=True, events=[f"safety:exception:{name}"])
        self._obs = obs
        # every object the success spec names must be seen (a multi-object spec has no `.object`)
        needed = success_objects(self.task.success)
        success = (
            self._arm.stopped is None
            and all(n in obs.objects for n in needed)
            and is_success(self.task, {n: s.pos for n, s in obs.objects.items()}, self._start)
        )
        return obs, StepInfo(success=bool(success), terminated=info.terminated, events=info.events)

    def randomization_sample(self) -> dict[str, float]:
        """The drawn placement and where the object was measured to be. A twin that really
        applies the physical draws reports its own sample; a real arm reports them as nominal."""
        if hasattr(self.robot, "randomization_sample"):
            sample = {**self.robot.randomization_sample(), "real.dry_run": 1.0}
        else:
            sample = {**self._arm.sample, **PHYSICAL_NOMINAL, "real.physical_nominal": 1.0}
        if hasattr(self.frames, "visual_sample"):
            sample.update(self.frames.visual_sample())
        return {**sample, **self._measured}

    def render(self, camera: str | None = None) -> np.ndarray | None:
        if self.frames is None:
            # a simulated twin can show itself; a real arm with no camera has nothing to show
            return self.robot.render(camera) if hasattr(self.robot, "render") else None
        frame = self.frames.capture(camera or self.cameras[0])
        return None if frame is None else frame.rgb

    def close(self) -> None:
        self._env.close()

    # -- real-robot extras -------------------------------------------------

    def stop(self, reason: str = "requested") -> None:
        """Stop the arm and end the episode. It stays stopped until the next `reset`."""
        self._arm.stop(reason)

    @property
    def stopped(self) -> str | None:
        return self._arm.stopped

    @property
    def safety_log(self) -> list[tuple[int, str]]:
        """(step, event) for every safety event since the last reset."""
        return list(self._arm.safety_log)

    @property
    def command(self) -> tuple[np.ndarray, float]:
        """The joint targets and gripper command last sent to the robot (for logs and system ID)."""
        return self._arm._q_cmd.copy(), self._arm._grip

    @property
    def targets(self) -> dict[str, Placement]:
        return dict(self._arm.targets)

    def _measure(self) -> dict[str, Estimate]:
        if self.estimator is None:
            return {}
        obs = self._arm.observe(self.robot.read_state())
        frames = []
        if self.estimator.needs_frames:
            frames = [f for f in (self.frames.capture(c) for c in self.cameras) if f is not None]
        return self.estimator.estimate(frames, obs)


__all__ = [
    "AutoOperator",
    "Controller",
    "Measure",
    "Operator",
    "Placement",
    "PlacementAborted",
    "RealEnv",
    "RobotIO",
    "RobotState",
    "draw_placement",
]
