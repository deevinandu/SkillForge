"""A kinematic domino world for testing ToppleChain's logic without physics (test helper only).

StubEnv plus: the EE has an orientation (it snaps to the commanded one); a closed gripper picks
the domino whose long axis passes within 2.5 cm of the fingertips (standing or lying) and carries
it rigidly (position and orientation); an opened one puts it down on whichever face points most
nearly down; a domino a nearly closed hand runs into while moving along its face normal falls over
that way, knocking over the next one in its path if it is within reach (its height) and roughly in
line. It is a logic harness, not a model of the physics.
"""

from __future__ import annotations

import math

import numpy as np

from skillforge.contracts import Action, ObjectState, Observation
from skillforge.sim import StubEnv
from skillforge.sim.disturbances import knock_axis, quat_mul, yaw_quat
from skillforge.skills.dominoes import heading_x, tilt, up_of
from skillforge.skills.grasp import mat_to_quat, quat_to_mat

FALLEN_TILT = 1.3


class DominoKinematicEnv(StubEnv):
    def reset(self, seed: int) -> Observation:
        self._ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self._carry: tuple[np.ndarray, np.ndarray, float] | None = None
        self._new_falls: list[str] = []
        super().reset(seed)
        for o in self.task.scene.objects:  # the stub starts at identity; dominoes keep their spec yaw
            if self.task.scene.scatter is None or o.name not in (self.task.scene.scatter.objects or self._obj):
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
        R_h = quat_to_mat(self._ee_quat)
        if closed and not was and self._held is None and self._grip > 0.95:
            near = []  # the domino whose long axis passes closest to the fingertips
            for k, p in self._obj.items():
                size = np.asarray(self.task.scene.object(k).size, dtype=float)
                R = quat_to_mat(self._quat[k])
                L = R[:, int(np.argmax(size))]
                s_ = float(np.clip((self._ee - p) @ L, -size.max(), size.max()))
                near.append((float(np.linalg.norm(self._ee - (p + s_ * L))), k))
            dist, k = min(near) if near else (np.inf, None)
            if dist < 0.025:
                p, R = self._obj[k], quat_to_mat(self._quat[k])
                size = np.asarray(self.task.scene.object(k).size, dtype=float)
                self._held = k
                width = 2 * size[int(np.argmax(np.abs(R.T @ R_h[:, 1])))]
                self._carry = (R_h.T @ (p - self._ee), R_h.T @ R, width)
                events.append(f"grasp:{k}")
        if not closed and self._held is not None:
            k = self._held
            self._obj[k], self._quat[k] = self._rest(k)
            events.append(f"release:{k}")
            self._held, self._carry = None, None
        if self._held is not None:
            p_hd, R_hd, _ = self._carry
            self._obj[self._held] = self._ee + R_h @ p_hd
            self._quat[self._held] = mat_to_quat(R_h @ R_hd)
        self._push()
        self._cascade()
        return events

    def _rest(self, k: str) -> tuple[np.ndarray, np.ndarray]:
        """Put `k` down on the face that points most nearly down, keeping its heading."""
        size = np.asarray(self.task.scene.object(k).size, dtype=float)
        R = quat_to_mat(self._quat[k])
        v = int(np.argmax(np.abs(R[2, :])))
        j = int(np.argmax(size)) if v != int(np.argmax(size)) else 0
        out = np.zeros((3, 3))
        out[:, v] = [0.0, 0.0, 1.0 if R[2, v] > 0 else -1.0]
        h = R[:, j].copy()
        h[2] = 0.0
        out[:, j] = h / max(float(np.linalg.norm(h)), 1e-9)
        i = 3 - v - j
        out[:, i] = np.cross(out[:, (i + 1) % 3], out[:, (i + 2) % 3])
        pos = self._obj[k].copy()
        floor = 0.0  # or the top of a fixture (shelf board) under it, if it was let go above one
        for f in self.task.scene.fixtures:
            c, h = np.asarray(f.center), np.asarray(f.half_extents)
            if bool(np.all(np.abs(pos[:2] - c[:2]) <= h[:2])) and c[2] + h[2] <= pos[2] + 0.01:
                floor = max(floor, float(c[2] + h[2]))
        pos[2] = floor + size[v]
        return pos, mat_to_quat(out)

    def _fall(self, k: str, heading: float) -> None:
        ax = knock_axis(heading)
        q = np.concatenate([[math.cos(FALLEN_TILT / 2)], math.sin(FALLEN_TILT / 2) * ax])
        self._quat[k] = quat_mul(q, yaw_quat(heading_x(self._quat[k])))
        self._new_falls.append(k)

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
        """Only a domino that has just fallen (pushed, knocked, or hit by one) can hit the next."""
        while self._new_falls:
            k = self._new_falls.pop(0)
            p = self._obj[k]
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

    def _apply_disturbances(self, events):
        out = super()._apply_disturbances(events)
        self._new_falls += [e.rsplit(":", 1)[1] for e in out if e.startswith("disturbance:knock:")]
        self._cascade()
        return out

    def _observe(self) -> Observation:
        obs = super()._observe()
        obs.ee_quat = self._ee_quat.copy()
        if self._held is not None:
            obs.gripper_width = float(self._carry[2])
        obs.objects = {k: ObjectState(pos=o.pos, quat=self._quat[k].copy()) for k, o in obs.objects.items()}
        return obs
