"""PushToRegion v0.1: slide an object the gripper cannot grasp into a target region.

The hand pushes with the fingers half open and the closing axis across the push direction, so the
two finger pads touch the object's face about 5 cm apart: a two-point push that resists the object
spinning away. Each step the push direction is re-aimed from the object's observed position to the
region centre and the pusher is placed just behind the object on that line (closed loop). If the
object drifts sideways off the pusher, the hand lifts, re-plans and comes back down behind it.

States: plan -> above (pre-push point, at hover height) -> down -> push -> retreat -> done; from\ndone it re-plans if the (filtered) object position is outside the region.\n\nv0.2 (from tracing stress x2 timeouts): the object's xy is smoothed before use (1 cm observation\nnoise made it declare success early), it keeps checking after retreating, and it pushes below\nmid-height (high friction or mass tipped the box).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, Observation, TaskSpec
from skillforge.skills.grasp import DOWN, quat_angle, quat_from_axis_angle, quat_mul, yaw_of


class PushToRegion:
    name = "push_to_region"
    version = "0.2.0"

    hover = 0.10  # metres above the object's top for moves to and from the pre-push point
    standoff = 0.04  # metres between the pusher and the object's face at the pre-push point
    lead = 0.03  # metres the commanded pusher position runs ahead of the contact point
    contact_gap = 0.004  # metres behind the face the pusher aims for (keeps it in contact)
    max_offline = 0.025  # metres the object may drift off the push line before re-approaching
    done_margin = 0.012  # metres inside the region edge that counts as done (observations are noisy)
    tol = 0.008  # metres: "arrived" for the free-space moves
    gripper = 0.4  # half open: finger pads about 5 cm apart (> 0.5 would read as closed)
    max_replans = 6
    push_height = 0.4  # push this fraction of the half-height below the object's centre (less tipping)
    smooth = 0.3  # weight of a new observation in the exponential filter on the object's xy
    settle_steps = 6  # steps to watch the object after retreating before calling it done

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PushToRegion needs an object_in_region task")
        self.task = task
        self.spec = task.scene.object(task.success.object)
        region = task.scene.region(task.success.region)
        self.goal = np.asarray(region.center, float)[:2]
        self.half = np.asarray(region.half_extents, float)[:2]
        sz = np.asarray(self.spec.size, float)
        self.half_xy = sz[:2] if self.spec.shape == "box" else np.array([sz[0], sz[0]])
        self.half_z = {"box": sz[2], "cylinder": sz[1], "sphere": sz[0]}.get(self.spec.shape, sz[2])
        self.state = "plan"
        self.quat = DOWN.copy()
        self.replans = 0
        self.park: np.ndarray | None = None
        self.xy: np.ndarray | None = None  # filtered object xy (observations are noisy)
        self.watch = 0

    # -- geometry --------------------------------------------------------------

    def _dir(self, obj_xy: np.ndarray) -> np.ndarray:
        d = self.goal - obj_xy
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else np.array([0.0, 1.0])

    def _extent(self, u: np.ndarray, obj_quat: np.ndarray) -> float:
        """Half-extent of the object's footprint along `u`."""
        if self.spec.shape != "box":
            return float(self.half_xy[0])
        a = float(np.arctan2(u[1], u[0])) - yaw_of(obj_quat)
        return float(self.half_xy[0] * abs(np.cos(a)) + self.half_xy[1] * abs(np.sin(a)))

    def _push_quat(self, u: np.ndarray, ee_quat: np.ndarray) -> np.ndarray:
        """Top-down orientation with the closing axis across `u`; of the two, the one nearer the wrist."""
        psi = float(np.arctan2(u[1], u[0]))
        cands = [quat_mul(quat_from_axis_angle([0, 0, 1], psi + k * np.pi), DOWN) for k in (0, 1)]
        return min(cands, key=lambda q: quat_angle(q, ee_quat))

    def _in_region(self, obj_xy: np.ndarray) -> bool:
        return bool(np.all(np.abs(obj_xy - self.goal) <= self.half - self.done_margin))

    # -- policy ----------------------------------------------------------------

    def _go(self, pos, grip: float | None = None) -> Action:
        return Action(ee_pos=np.asarray(pos, float), ee_quat=self.quat, gripper=self.gripper if grip is None else grip)

    def act(self, obs: Observation) -> Action:
        o = obs.objects.get(self.spec.name)
        if o is None:
            return self._go(obs.ee_pos)
        obj = np.asarray(o.pos, float)
        self.xy = obj[:2].copy() if self.xy is None else (1 - self.smooth) * self.xy + self.smooth * obj[:2]
        obj = np.r_[self.xy, obj[2]]
        z_push = obj[2] - self.push_height * self.half_z  # below mid-height: less tipping
        z_high = obj[2] + self.half_z + self.hover
        if self.state not in ("retreat", "done") and self._in_region(obj[:2]):
            self.state = "retreat"

        if self.state == "plan":
            if self.replans > self.max_replans:
                self.state = "retreat"
            else:
                u = self._dir(obj[:2])
                self.u = u
                self.quat = self._push_quat(u, obs.ee_quat)
                self.pre = obj[:2] - u * (self._extent(u, o.quat) + self.standoff)
                self.replans += 1
                self.state = "above"
        if self.state == "above":
            tgt = np.r_[self.pre, z_high]
            if np.linalg.norm(obs.ee_pos - tgt) < self.tol:
                self.state = "down"
            else:
                return self._go(tgt)
        if self.state == "down":
            tgt = np.r_[self.pre, z_push]
            if np.linalg.norm(obs.ee_pos - tgt) < self.tol:
                self.state = "push"
            else:
                return self._go(tgt)
        if self.state == "push":
            u = self._dir(obj[:2])
            # lateral offset of the object from the line the pusher is driving along
            rel = obj[:2] - obs.ee_pos[:2]
            off = abs(float(rel @ np.array([-self.u[1], self.u[0]])))
            if off > self.max_offline or float(rel @ self.u) < 0:
                self.state = "lift"
            else:
                self.u = u
                contact = obj[:2] - u * (self._extent(u, o.quat) - self.contact_gap)
                ahead = min(self.lead, float(np.linalg.norm(self.goal - obj[:2])))
                return self._go(np.r_[contact + u * ahead, z_push])
        if self.state == "lift":
            tgt = np.r_[obs.ee_pos[:2], z_high]
            if obs.ee_pos[2] > z_high - self.tol:
                self.state = "plan"
                return self.act(obs)
            return self._go(tgt)
        if self.state == "retreat":
            if self.park is None:
                self.park = np.r_[obs.ee_pos[:2] - self.u * 0.03, z_high] if hasattr(self, "u") else np.r_[obs.ee_pos[:2], z_high]
            if np.linalg.norm(obs.ee_pos - self.park) < self.tol:
                self.state, self.watch = "done", 0
            return self._go(self.park)
        # done: keep watching; if the filtered position says the object is outside, push again
        self.watch += 1
        if self.watch > self.settle_steps and self.replans <= self.max_replans and not self._in_region(obj[:2]):
            self.park = None
            self.state = "plan"
            return self.act(obs)
        return self._go(self.park if self.park is not None else obs.ee_pos)
