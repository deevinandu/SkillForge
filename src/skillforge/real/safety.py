"""T7: safety limits, enforced by the env and never by the skill.

Agent-written skills only return `Action`s. `RealEnv` passes every action through `Safety`
before anything reaches the robot:

- workspace: an ee target outside the box is clamped to it (`safety:workspace`); a joint target
  whose tool point would leave the box is refused and the arm holds;
- joint speed: a joint target further than `joint_speed * dt` from the last command is pulled
  back to that distance (`safety:joint_speed`). The ee speed limit lives in the controller's
  reference generator, which is how the sim limits it too;
- force: a contact force above `force` stops the arm for the rest of the episode (`safety:force`);
- watchdog: a gap of more than `watchdog_periods` control periods between actions is reported
  (`safety:watchdog`) and the reference is re-synchronised to the measured state.

Each event is emitted once per streak, not once per step.

Nothing in this file talks to hardware and nothing here has run on a real arm.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import numpy as np

from skillforge.contracts import Action, Observation, Skill, TaskSpec


@dataclass(frozen=True)
class SafetyLimits:
    ee_speed: float = 0.1  # m/s. The brief: 0.1 to start, 0.25 only after 10 clean trials. Sim uses 0.5.
    ee_ang_speed: float = 1.0  # rad/s
    joint_speed: float = 1.0  # rad/s per joint
    force: float = 20.0  # N at the end effector; above this the arm stops
    watchdog_periods: float = 2.0  # control periods without an action before the arm holds


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned box in the world frame that the tool point must stay inside."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p: np.ndarray) -> bool:
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), self.lo, self.hi)

    @classmethod
    def from_task(
        cls,
        task: TaskSpec,
        *,
        margin: float = 0.10,  # metres added around everything the task mentions
        z_min: float = 0.005,  # metres above the table (z = 0): the tool point never goes lower
        z_max: float = 0.60,
        include: Iterable[np.ndarray] = (),  # extra points that must be inside, e.g. the home tool point
    ) -> Workspace:
        """The box around every object's possible start position and every region, plus a margin."""
        j = task.scene.randomization.object_xy_jitter.hi
        pts = []
        for o in task.scene.objects:
            p = np.asarray(o.pose.pos, dtype=float)[:2]
            pts += [p - j, p + j]
        for r in task.scene.regions:
            c, h = np.asarray(r.center, dtype=float)[:2], np.asarray(r.half_extents, dtype=float)[:2]
            pts += [c - h, c + h]
        extra = [np.asarray(p, dtype=float) for p in include]
        pts += [p[:2] for p in extra]
        if not pts:
            raise ValueError("cannot derive a workspace: the task has no objects or regions")
        xy = np.array(pts)
        top = max([z_max] + [float(p[2]) + margin for p in extra])
        lo = (*(xy.min(axis=0) - margin), z_min)
        hi = (*(xy.max(axis=0) + margin), top)
        return cls(tuple(float(v) for v in lo), tuple(float(v) for v in hi))


class Watchdog:
    """Trips when `feed()` has not been called for `timeout` seconds. The ROS layer runs one of
    these on its own timer so the arm holds even when the Python side is stuck; `RealEnv` uses
    one to notice the gap afterwards."""

    def __init__(self, timeout: float, clock: Callable[[], float]):
        self.timeout = timeout
        self.clock = clock
        self._last: float | None = None

    def feed(self) -> None:
        self._last = self.clock()

    def expired(self) -> bool:
        return self._last is not None and self.clock() - self._last > self.timeout


class Safety:
    def __init__(self, limits: SafetyLimits, workspace: Workspace):
        self.limits = limits
        self.workspace = workspace
        self._active: set[str] = set()

    def reset(self) -> None:
        self._active = set()

    def _report(self, name: str, on: bool) -> list[str]:
        """`name` once when the condition starts; again only after it has cleared."""
        if on and name not in self._active:
            self._active.add(name)
            return [name]
        if not on:
            self._active.discard(name)
        return []

    def check_action(self, action: Action, fk: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]) -> tuple[Action | None, list[str]]:
        """The action to execute (None = hold the last command) and any safety events."""
        if action.mode == "ee":
            inside = self.workspace.contains(action.ee_pos)
            events = self._report("safety:workspace", not inside)
            return (action if inside else replace(action, ee_pos=self.workspace.clamp(action.ee_pos))), events
        pos, _ = fk(np.asarray(action.joint_targets, dtype=float))
        inside = self.workspace.contains(pos)
        return (action if inside else None), self._report("safety:workspace", not inside)

    def limit_joints(self, q_des: np.ndarray, q_prev: np.ndarray, dt: float) -> tuple[np.ndarray, list[str]]:
        dq = q_des - q_prev
        step = self.limits.joint_speed * dt
        over = bool(np.any(np.abs(dq) > step))
        events = self._report("safety:joint_speed", over)
        # scale the whole step, so the joints stay coordinated and the tool keeps its direction
        return (q_prev + dq * (step / np.max(np.abs(dq))) if over else q_des), events

    def force_exceeded(self, force: np.ndarray | None) -> bool:
        return force is not None and float(np.linalg.norm(force)) > self.limits.force


class GuardedSkill:
    """Wraps a skill so that an exception in `act` stops the arm before it propagates.

    The harness catches a crashing skill and records the episode as `exception`, but it does not
    tell the env. On hardware the arm must be stopped first.
    """

    def __init__(self, skill: Skill, env):
        self.skill = skill
        self.env = env
        self.name = skill.name
        self.version = skill.version

    def reset(self, task: TaskSpec) -> None:
        self.skill.reset(task)

    def act(self, obs: Observation) -> Action:
        try:
            return self.skill.act(obs)
        except Exception as e:
            self.env.stop(f"exception:{type(e).__name__}")
            raise
