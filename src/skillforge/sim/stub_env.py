"""A kinematic toy env: the end effector is a point, objects are carried when gripped.

It exists so T2/T3/T4 can build and test against the SimEnv contract before T1's MuJoCo
env lands. It honours the randomization fields that make sense without physics
(xy/yaw jitter, observation noise, latency) and records the rest as sampled values.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from skillforge.contracts import Action, ObjectState, Observation, StepInfo, TaskSpec

from .success import is_success

TABLE_Z = 0.0
GRASP_RADIUS = 0.02  # metres: EE must be this close to an object's centre to grip it
MAX_EE_SPEED = 0.5  # m/s


class StubEnv:
    def __init__(self, task: TaskSpec):
        self.task = task
        self.dt = 1.0 / task.control_hz
        self._rng = np.random.default_rng(0)
        self._sample: dict[str, float] = {}
        self.reset(0)

    # -- contract ----------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed)
        rnd = self.task.scene.randomization
        u = lambda r: float(self._rng.uniform(r.lo, r.hi))
        self._sample = {
            "object_xy_jitter": u(rnd.object_xy_jitter),
            "object_yaw_jitter": u(rnd.object_yaw_jitter),
            "friction_scale": u(rnd.friction_scale),
            "mass_scale": u(rnd.mass_scale),
            "actuator_gain_scale": u(rnd.actuator_gain_scale),
            "obs_noise_std": u(rnd.obs_noise_std),
            "action_latency_steps": float(round(u(rnd.action_latency_steps))),
        }
        j = self._sample["object_xy_jitter"]
        self._obj = {}
        for o in self.task.scene.objects:
            p = np.array(o.pose.pos, dtype=float)
            if not o.fixed:
                p[:2] += self._rng.uniform(-j, j, size=2)
            self._obj[o.name] = p
        self._start = {k: v.copy() for k, v in self._obj.items()}
        self._ee = np.array(self.task.scene.robot.base_pose.pos, dtype=float) + np.array([0.4, 0.0, 0.3])
        self._grip = 0.0
        self._held: str | None = None
        self._t = 0.0
        self._queue: deque[Action] = deque()
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        self._queue.append(action)
        lag = int(self._sample["action_latency_steps"])
        act = self._queue.popleft() if len(self._queue) > lag else None
        events: list[str] = []
        if act is not None:
            events += self._apply(act)
        self._t += self.dt
        info = StepInfo(
            success=is_success(self.task, self._obj, self._start),
            terminated=any(p[2] < TABLE_Z - 0.05 for p in self._obj.values()),
            events=events,
        )
        return self._observe(), info

    def randomization_sample(self) -> dict[str, float]:
        return dict(self._sample)

    def render(self, camera: str | None = None) -> np.ndarray | None:
        return None

    def close(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _apply(self, act: Action) -> list[str]:
        events = []
        if act.mode == "ee":
            delta = np.asarray(act.ee_pos, dtype=float) - self._ee
            step = MAX_EE_SPEED * self.dt * self._sample["actuator_gain_scale"]
            n = np.linalg.norm(delta)
            self._ee = self._ee + (delta if n <= step else delta / n * step)
            self._ee[2] = max(self._ee[2], TABLE_Z)
        was_closed = self._grip > 0.5
        self._grip = act.gripper
        closed = self._grip > 0.5
        if closed and not was_closed and self._held is None:
            near = [k for k, p in self._obj.items() if np.linalg.norm(p - self._ee) < GRASP_RADIUS]
            if near:
                self._held = near[0]
                events.append(f"grasp:{near[0]}")
            else:
                events.append("grasp_miss")
        if not closed and self._held is not None:
            events.append(f"release:{self._held}")
            self._held = None
        if self._held is not None:
            self._obj[self._held] = self._ee.copy()
        else:
            for k, p in self._obj.items():  # unsupported objects drop back to their resting height
                p[2] = self._start[k][2]
        return events

    def _observe(self) -> Observation:
        sd = self._sample["obs_noise_std"]
        objs = {
            k: ObjectState(pos=p + self._rng.normal(0, sd, 3) if sd > 0 else p.copy(), quat=np.array([1.0, 0, 0, 0]))
            for k, p in self._obj.items()
        }
        return Observation(
            t=self._t,
            qpos=self._ee.copy(),
            qvel=np.zeros(3),
            ee_pos=self._ee.copy(),
            ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
            gripper_width=self.task.scene.robot.max_gripper_width * (1.0 - self._grip),
            objects=objs,
        )
