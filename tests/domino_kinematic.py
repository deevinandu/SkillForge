"""A kinematic domino world for testing ToppleChain's logic without physics (test helper only).

StubEnv plus: the EE has an orientation (it snaps to the commanded one), a closed gripper picks a
standing domino under the fingertips and carries it rigidly (position and yaw), an opened one
stands it back on the table, and a domino a nearly closed hand runs into while moving along its
face normal falls over that way, knocking over the next one in its path if it is within reach (its
height) and roughly in line. It is a logic harness, not a model of the physics.
"""

from __future__ import annotations

import math

import numpy as np

from skillforge.contracts import Action, ObjectState, Observation
from skillforge.sim import StubEnv
from skillforge.sim.disturbances import knock_axis, quat_mul, yaw_quat
from skillforge.skills.dominoes import heading_x, tilt, up_of

FALLEN_TILT = 1.3


class DominoKinematicEnv(StubEnv):
    def reset(self, seed: int) -> Observation:
        self._ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self._carry: tuple[np.ndarray, float] | None = None
        super().reset(seed)
        for o in self.task.scene.objects:  # the stub starts at identity; dominoes keep their spec yaw
            self._quat[o.name] = np.asarray(o.pose.quat, dtype=float)
        self._prev_ee = self._ee.copy()
        return self._observe()

    # -- kinematics ------------------------------------------------------------

    def _apply(self, act: Action) -> list[str]:
        events = []
        self._prev_ee = self._ee.copy()
        if act.mode == "ee":
            delta = np.asarray(act.ee_pos, dtype=float) - self._ee
            step = 0.5 * self.dt * self._sample["actuator_gain_scale"]
            n = float(np.linalg.norm(delta))
            self._ee = self._ee + (delta if n <= step else delta / n * step)
            if act.ee_quat is not None:
                self._ee_quat = np.asarray(act.ee_quat, dtype=float) / np.linalg.norm(act.ee_quat)
        was, self._grip = self._grip > 0.5, act.gripper
        closed = self._grip > 0.5
        hand_yaw = heading_x(self._ee_quat)
        if closed and not was and self._held is None and self._grip > 0.95:
            for k, p in self._obj.items():
                h = 2 * self.task.scene.object(k).size[2]
                if tilt(self._quat[k]) < 0.2 and np.linalg.norm(p[:2] - self._ee[:2]) < 0.02 and 0.0 < self._ee[2] < h:
                    self._held = k
                    self._carry = (p - self._ee, heading_x(self._quat[k]) - hand_yaw, hand_yaw)
                    events.append(f"grasp:{k}")
                    break
        if not closed and self._held is not None:
            k = self._held
            self._obj[k][2] = self.task.scene.object(k).size[2]  # stands on the table
            events.append(f"release:{k}")
            self._held, self._carry = None, None
        if self._held is not None:
            off, dyaw, yaw0 = self._carry
            turn = hand_yaw - yaw0
            c, s = math.cos(turn), math.sin(turn)
            off = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]])
            self._obj[self._held] = self._ee + off
            self._quat[self._held] = yaw_quat(hand_yaw + dyaw)
        self._push()
        self._cascade()
        return events

    def _fall(self, k: str, heading: float) -> None:
        ax = knock_axis(heading)
        q = np.concatenate([[math.cos(FALLEN_TILT / 2)], math.sin(FALLEN_TILT / 2) * ax])
        self._quat[k] = quat_mul(q, yaw_quat(heading_x(self._quat[k])))

    def _push(self) -> None:
        # Only a nearly closed hand pushes (ToppleChain pushes with ~8 mm between the fingers); an
        # open hand coming down around a domino to grasp it does not knock it over here.
        v = self._ee - self._prev_ee
        if not 0.5 < self._grip < 0.95 or float(np.linalg.norm(v[:2])) < 1e-5:
            return
        for k, p in self._obj.items():
            if k == self._held or tilt(self._quat[k]) > 0.2:
                continue
            sz = self.task.scene.object(k).size
            nrm = np.array([math.cos(heading_x(self._quat[k])), math.sin(heading_x(self._quat[k]))])
            rel = self._ee[:2] - p[:2]
            along, lat = float(rel @ nrm), float(rel @ np.array([-nrm[1], nrm[0]]))
            if abs(along) < sz[0] + 0.012 and abs(lat) < sz[1] and 0.0 < self._ee[2] < 2 * sz[2]:
                push = float(v[:2] @ nrm)
                if abs(push) > 1e-5:
                    self._fall(k, math.atan2(*(nrm * np.sign(push))[::-1]))

    def _cascade(self) -> None:
        changed = True
        while changed:
            changed = False
            for k, p in self._obj.items():
                if tilt(self._quat[k]) < 1.0 or k == self._held:
                    continue
                top = up_of(self._quat[k])[:2]
                head = top / max(np.linalg.norm(top), 1e-9)
                reach = 2 * self.task.scene.object(k).size[2]
                for j, q in self._obj.items():
                    if j == k or j == self._held or tilt(self._quat[j]) > 0.2:
                        continue
                    rel = q[:2] - p[:2]
                    a = float(rel @ head)
                    if 0.0 < a < reach and abs(float(rel @ np.array([-head[1], head[0]]))) < 0.025:
                        self._fall(j, math.atan2(head[1], head[0]))
                        changed = True

    def _observe(self) -> Observation:
        obs = super()._observe()
        obs.ee_quat = self._ee_quat.copy()
        if self._held is not None:
            obs.gripper_width = 2 * self.task.scene.object(self._held).size[1]
        obs.objects = {k: ObjectState(pos=o.pos, quat=self._quat[k].copy()) for k, o in obs.objects.items()}
        return obs
