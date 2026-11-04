"""A scripted, closed-loop pick-and-place skill: the hand-written baseline that agents improve on.

Closed-loop means each phase re-reads the object's observed position every step,
so a jittered start or a nudged object is tracked rather than replayed blindly.
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, Observation, TaskSpec

DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down


class PickPlace:
    name = "pick_place"
    version = "0.1.0"

    hover = 0.10  # metres above the target before descending
    tol = 0.008  # metres: position error that counts as "arrived"
    settle_steps = 4  # steps to hold the gripper command before moving on

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PickPlace needs an object_in_region task")
        self.obj = task.success.object
        self.goal = np.asarray(task.scene.region(task.success.region).center, dtype=float)
        self.phase = "approach"
        self.hold = 0

    def act(self, obs: Observation) -> Action:
        o = obs.objects[self.obj].pos
        above_obj = o + [0, 0, self.hover]
        above_goal = self.goal + [0, 0, self.hover]
        at = lambda target: np.linalg.norm(obs.ee_pos - target) < self.tol

        if self.phase == "approach":
            if at(above_obj):
                self.phase = "descend"
            return Action(ee_pos=above_obj, ee_quat=DOWN, gripper=0.0)
        if self.phase == "descend":
            if at(o):
                self.phase, self.hold = "grip", 0
            return Action(ee_pos=o, ee_quat=DOWN, gripper=0.0)
        if self.phase == "grip":
            self.hold += 1
            if self.hold >= self.settle_steps:
                self.phase = "lift"
            return Action(ee_pos=obs.ee_pos, ee_quat=DOWN, gripper=1.0)
        if self.phase == "lift":
            target = obs.ee_pos.copy()
            target[2] = above_goal[2]
            if at(target):
                self.phase = "transport"
            return Action(ee_pos=target, ee_quat=DOWN, gripper=1.0)
        if self.phase == "transport":
            if at(above_goal):
                self.phase = "lower"
            return Action(ee_pos=above_goal, ee_quat=DOWN, gripper=1.0)
        if self.phase == "lower":
            if at(self.goal):
                self.phase, self.hold = "release", 0
            return Action(ee_pos=self.goal, ee_quat=DOWN, gripper=1.0)
        # release, then retreat
        self.hold += 1
        if self.hold >= self.settle_steps:
            return Action(ee_pos=above_goal, ee_quat=DOWN, gripper=0.0)
        return Action(ee_pos=obs.ee_pos, ee_quat=DOWN, gripper=0.0)
