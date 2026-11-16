"""T7: controllers. `MujocoController` is the sim's controller run on a kinematic model.

`MujocoEnv` (T1) owns the reference generator and the damped-least-squares IK and does not
expose them, so `MujocoController` builds a `MujocoEnv` for the task, never steps its physics,
and drives its private control methods. That keeps one controller for sim and real. It also
means this file breaks if T1 renames those methods: see the request to T1 in HANDOVER for a
public controller. Every private name used is in this file and nowhere else in `real/`.
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, TaskSpec

DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # tool pointing down, the stub robot's only orientation


class MujocoController:
    """`contracts.Action` -> joint targets, with `MujocoEnv`'s speed-limited reference and IK."""

    def __init__(self, task: TaskSpec, *, max_ee_speed: float = 0.1, max_ee_ang_speed: float = 1.0, **env_kwargs):
        from skillforge.sim.mujoco_env import MujocoEnv  # lazy: mujoco is an optional dependency

        self._env = MujocoEnv(task, max_ee_speed=max_ee_speed, max_ee_ang_speed=max_ee_ang_speed, **env_kwargs)

    @property
    def home(self) -> np.ndarray:
        return self._env._home.copy()

    def _restart(self, q: np.ndarray) -> None:
        e = self._env
        e._q_des = np.asarray(q, dtype=float).copy()
        e._fk(e._q_des)
        e._ref_pos = e._scratch.site_xpos[e._site].copy()
        e._ref_quat = e._site_quat(e._scratch)

    def reset(self, q: np.ndarray) -> None:
        self._restart(q)
        self._env._ik_failing = False

    def sync(self, q: np.ndarray) -> None:
        self._restart(q)

    def solve(self, action: Action) -> tuple[np.ndarray, list[str]]:
        events = self._env._control(action)
        return self._env._q_des.copy(), events

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = self._env
        e._fk(np.asarray(q, dtype=float))
        return e._scratch.site_xpos[e._site].copy(), e._site_quat(e._scratch)

    def close(self) -> None:
        self._env.close()


class PointController:
    """For the stub robot, whose three 'joints' are the tool point's x, y, z. No MuJoCo needed."""

    def __init__(self, dt: float, *, max_ee_speed: float = 0.1):
        self.dt = dt
        self.max_ee_speed = max_ee_speed
        self._ref = np.zeros(3)

    def reset(self, q: np.ndarray) -> None:
        self._ref = np.asarray(q, dtype=float)[:3].copy()

    sync = reset

    def solve(self, action: Action) -> tuple[np.ndarray, list[str]]:
        if action.mode == "joint":
            self._ref = np.asarray(action.joint_targets, dtype=float)[:3].copy()
        else:
            delta = np.asarray(action.ee_pos, dtype=float) - self._ref
            n, step = float(np.linalg.norm(delta)), self.max_ee_speed * self.dt
            self._ref = self._ref + (delta if n <= step else delta / n * step)
        return self._ref.copy(), []

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(q, dtype=float)[:3].copy(), DOWN.copy()
