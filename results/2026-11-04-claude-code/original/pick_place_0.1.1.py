"""PickPlace v0.1.1 (superseded by v0.2 in pick_place.py; kept as a baseline and for regression).

Closed-loop means each phase re-reads the object's observed position every step,
so a jittered start or a nudged object is tracked rather than replayed blindly.

Over v0.1.0:
- the gripper is yawed to the cube's observed yaw (mod 90 deg), so the fingers close on two
  flat faces instead of on edges/corners of a rotated cube (the main source of grasp_slip);
- the observed object position is low-pass filtered, so noise does not jerk the descent;
- if the cube leaves the gripper while lifting/carrying, the skill re-grasps it (bounded).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, Observation, TaskSpec
from skillforge.skills.grasp import DOWN, quat_from_axis_angle, quat_mul, yaw_of


def _wrap_quarter(yaw: float) -> float:
    """Reduce a yaw to [-pi/4, pi/4): a cube looks the same every 90 degrees."""
    q = np.pi / 2
    return float((yaw + q / 2) % q - q / 2)


class PickPlaceV01:
    name = "pick_place"
    version = "0.1.1"

    hover = 0.10  # metres above the target before descending
    tol = 0.008  # metres: position error that counts as "arrived"
    settle_steps = 6  # steps to hold the gripper command before moving on
    alpha = 0.35  # low-pass gain on observed object position
    slip_dist = 0.04  # metres: object farther than this from the TCP while held = lost
    max_regrasps = 3
    carry_step = 0.04  # metres: max lead of the commanded target over the EE while holding

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PickPlace needs an object_in_region task")
        self.obj = task.success.object
        self.goal = np.asarray(task.scene.region(task.success.region).center, dtype=float)
        self.phase = "approach"
        self.hold = 0
        self.o = None
        self.quat = DOWN.copy()
        self.regrasps = 0

    def _gentle(self, obs: Observation, target: np.ndarray) -> np.ndarray:
        """Cap how far ahead of the EE the target is while carrying, to limit accelerations."""
        d = np.asarray(target, dtype=float) - obs.ee_pos
        n = float(np.linalg.norm(d))
        return obs.ee_pos + d * (self.carry_step / n) if n > self.carry_step else np.asarray(target, dtype=float)

    def _grasp_quat(self, obs: Observation) -> np.ndarray:
        yaw = _wrap_quarter(yaw_of(np.asarray(obs.objects[self.obj].quat, dtype=float)))
        return quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], yaw), DOWN)

    def act(self, obs: Observation) -> Action:
        raw = np.asarray(obs.objects[self.obj].pos, dtype=float)
        if self.o is None or np.linalg.norm(raw - self.o) > 0.03:
            self.o = raw.copy()  # first sight, or the object really moved: snap
        else:
            self.o = self.o + self.alpha * (raw - self.o)
        o = self.o
        above_obj = o + [0, 0, self.hover]
        above_goal = self.goal + [0, 0, self.hover]
        at = lambda target: np.linalg.norm(obs.ee_pos - target) < self.tol
        lost = lambda: np.linalg.norm(raw - obs.ee_pos) > self.slip_dist

        if self.phase in ("lift", "transport", "lower") and lost() and self.regrasps < self.max_regrasps:
            self.regrasps += 1
            self.phase = "reopen"
            self.hold = 0

        if self.phase == "reopen":
            # open and rise straight up before approaching the object again
            self.hold += 1
            up = obs.ee_pos.copy()
            up[2] = max(up[2], o[2] + self.hover)
            if self.hold >= self.settle_steps and at(up):
                self.phase = "approach"
            return Action(ee_pos=up, ee_quat=self.quat, gripper=0.0)
        if self.phase == "approach":
            self.quat = self._grasp_quat(obs)
            if at(above_obj):
                self.phase = "descend"
            return Action(ee_pos=above_obj, ee_quat=self.quat, gripper=0.0)
        if self.phase == "descend":
            if at(o):
                self.phase, self.hold = "grip", 0
            return Action(ee_pos=o, ee_quat=self.quat, gripper=0.0)
        if self.phase == "grip":
            self.hold += 1
            if self.hold >= self.settle_steps:
                self.phase = "lift"
            return Action(ee_pos=obs.ee_pos, ee_quat=self.quat, gripper=1.0)
        if self.phase == "lift":
            target = obs.ee_pos.copy()
            target[2] = above_goal[2]
            if at(target):
                self.phase = "transport"
            return Action(ee_pos=self._gentle(obs, target), ee_quat=self.quat, gripper=1.0)
        if self.phase == "transport":
            if at(above_goal):
                self.phase = "lower"
            return Action(ee_pos=self._gentle(obs, above_goal), ee_quat=self.quat, gripper=1.0)
        if self.phase == "lower":
            if at(self.goal):
                self.phase, self.hold = "release", 0
            return Action(ee_pos=self._gentle(obs, self.goal), ee_quat=self.quat, gripper=1.0)
        # release, then retreat
        self.hold += 1
        if self.hold >= self.settle_steps:
            return Action(ee_pos=above_goal, ee_quat=self.quat, gripper=0.0)
        return Action(ee_pos=obs.ee_pos, ee_quat=self.quat, gripper=0.0)
