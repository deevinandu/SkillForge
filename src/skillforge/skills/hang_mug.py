"""HangMug v0.1: pick a mug off the table and hang it by its handle on a peg of a mug stand.

The plan runs backwards from the hanging pose:
1. Hanging pose. The peg must pass through the handle loop, whose hole runs along the mug's y axis
   (the handle is on the mug's +x side). So the mug is turned until its y axis lies along the peg's
   horizontal direction. That leaves two yaws (handle to the left or to the right of the peg);
   `_choose_yaw` takes the one that keeps the wrist furthest from its joint limit.
2. Grasp. A rim pinch on the far side from the handle: one finger inside the cup, one outside,
   closing along the handle direction, `grasp_depth` below the rim. The handle stays free for the
   peg, and the hand, which is thin (+-32 mm) across its closing axis, faces the stand edge-on.
3. Threading. The grasp fixes the mug in the hand, so every later target is a MUG pose, turned into
   an EE pose with the mug-to-hand transform measured right after the grasp (averaged over a few
   steps): mug upright, loop centre on the peg's axis beyond its tip (`pre_s`), then slowly along
   the axis to `thread_s` (fraction of the peg from the post; far enough out that the hand clears
   the post), then down until the handle's top bar sits on the peg.
4. Release and retreat: open, rise straight up (the inner finger leaves the cup), back away along
   the peg, and wait for the mug to settle.

Robot-specific numbers are class attributes (`grasp_depth`, `thread_s`, `wrist_home_yaw`,
`wrist_limit`, speeds), so a fix that only changes them is a fix for this arm. Imports only
`skillforge.*` and numpy, so it loads as a `SkillSource`.
"""

from __future__ import annotations

import math

import numpy as np

from skillforge.contracts import Action, Observation, TaskSpec, success_conditions
from skillforge.sim.mug import handle_opening, peg_segment
from skillforge.skills.grasp import quat_to_mat


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _down_hand(beta: float) -> np.ndarray:
    """Gripper pointing down with its closing axis (local y) at world yaw `beta`."""
    y = np.array([math.cos(beta), math.sin(beta), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


class HangMug:
    name = "hang_mug"
    version = "0.1.0"

    # grasp (robot-specific: finger length and pad size)
    grasp_depth = 0.018  # metres from the rim top down to the fingertip centre
    open_cmd = 0.5  # gripper command while straddling the wall (0.5 = 40 mm on the Franka Hand)
    hover = 0.08  # metres above the grasp before descending
    grip_steps = 10
    measure_steps = 6  # steps averaged for the mug-to-hand transform after lifting
    lift = 0.06
    # threading (robot-specific: hand size against the stand)
    pre_s = 1.4  # loop centre this far along the peg (1 = tip) before threading
    thread_s = 0.68  # ... and here when threaded (the hand must clear the post)
    clear_above = 0.06  # metres the mug's bottom passes above the peg tip during the carry
    seat_drop = 0.004  # metres below 'top bar touching the peg' to lower before letting go
    # motion
    free_speed = 0.25  # m/s
    carry_speed = 0.15
    thread_speed = 0.03
    tol = 0.004
    # wrist (robot-specific: Franka joint 7 against the down-pointing hand)
    wrist_home_yaw = -math.pi / 2  # closing-axis yaw of the home orientation
    wrist_limit = 2.0  # rad of closing-axis turn from home we allow
    track_alpha = 0.2  # low-pass gain on the measured mug-in-hand position
    drift_tol = 0.03  # metres of mug-to-hand drift that means the mug slipped
    max_regrasps = 2

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        leaf = next((c for c in success_conditions(task.success) if c.kind == "object_hung"), None)
        if leaf is None:
            raise ValueError("HangMug needs an object_hung success condition")
        self.obj = leaf.object
        self.mug = task.scene.object(self.obj)
        self.peg = task.scene.object(leaf.support)
        self.r_out, self.half_h, self.wall = self.mug.size
        x0, x1, z0, z1 = handle_opening(self.mug)
        self.loop_c = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])  # loop centre, mug frame
        self.loop_top = z1
        root, tip = peg_segment(self.peg)
        self.root, self.tip = root, tip
        self.axis = (tip - root) / np.linalg.norm(tip - root)
        self.peg_r = float(self.peg.size[0])
        self.phase = "plan"
        self.k = 0
        self.ref: np.ndarray | None = None
        self.quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.grip = 0.0
        self.regrasps = 0
        self.rel: list[np.ndarray] = []
        self.log: list[str] = []

    # ---------------------------------------------------------------- geometry

    def _choose_yaw(self, psi: float) -> tuple[float, float]:
        """(hand closing-axis yaw at the grasp, mug yaw when hanging). The closing axis is along the
        handle (either sign); the hanging yaw puts the mug's y along the peg (either sign). Pick the
        pair whose wrist turn, measured continuously from home, stays smallest."""
        dh = math.atan2(self.axis[1], self.axis[0])
        best = None
        for sign in (0.0, math.pi):
            b0 = self.wrist_home_yaw + _wrap(psi + sign - self.wrist_home_yaw)
            for side in (-math.pi / 2, math.pi / 2):
                psi_h = dh + side
                b1 = b0 + _wrap(psi_h - psi)
                cost = max(abs(b0 - self.wrist_home_yaw), abs(b1 - self.wrist_home_yaw))
                if best is None or cost < best[0]:
                    best = (cost, b0, psi_h)
        if best[0] > self.wrist_limit:
            self.log.append(f"wrist turn {best[0]:.2f} rad over limit {self.wrist_limit}")
        return best[1], best[2]

    def _ee_for_mug(self, mug_pos: np.ndarray, mug_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """EE pose that puts the mug at (mug_pos, mug_rot), via the measured mug-in-hand transform."""
        p_rel, r_rel = self.T_rel  # mug pose in the hand frame
        r_e = mug_rot @ r_rel.T
        return mug_pos - r_e @ p_rel, _mat_to_quat(r_e)

    def _mug_at_s(self, s: float, dz: float = 0.0) -> np.ndarray:
        """Mug centre that puts the loop centre on the peg axis at fraction s (0 root, 1 tip)."""
        p = self.root + s * (self.tip - self.root)
        return p - self.R_hang @ self.loop_c + np.array([0.0, 0.0, dz])

    # ---------------------------------------------------------------- motion

    def _goto(self, obs: Observation, goal: np.ndarray, speed: float) -> bool:
        """Advance a reference toward goal at `speed`; True once the EE is there."""
        if self.ref is None:
            self.ref = obs.ee_pos.copy()
        d = goal - self.ref
        n = float(np.linalg.norm(d))
        step = speed / self.task.control_hz
        self.ref = goal.copy() if n <= step else self.ref + d * (step / n)
        return n <= step and float(np.linalg.norm(obs.ee_pos - goal)) < self.tol

    def _next(self, phase: str) -> None:
        self.log.append(f"{self.phase} -> {phase}")
        self.phase, self.k = phase, 0

    def _act(self) -> Action:
        return Action(mode="ee", ee_pos=self.ref.copy(), ee_quat=self.quat.copy(), gripper=self.grip)

    # ---------------------------------------------------------------- policy

    def act(self, obs: Observation) -> Action:
        self.k += 1
        mo = obs.objects[self.obj]
        m_pos, m_rot = np.asarray(mo.pos, dtype=float), quat_to_mat(mo.quat)
        if self.ref is None:
            self.ref = obs.ee_pos.copy()
            self.quat = np.asarray(obs.ee_quat, dtype=float).copy()

        if self.phase in ("carry", "turn", "pre", "thread", "seat"):
            self._track(obs, m_pos, m_rot)
        if self.phase in ("carry", "turn", "pre", "thread", "seat") and self._slipped(obs, m_pos):
            if self.regrasps < self.max_regrasps:
                self.regrasps += 1
                self.grip = 0.0
                self._next("back_off")
            else:
                self._next("give_up")

        if self.phase == "plan":
            if m_rot[2, 2] < 0.9:
                self._next("give_up")  # mug is not upright: this skill cannot pick it
                return self._act()
            psi = math.atan2(m_rot[1, 0], m_rot[0, 0])  # handle direction
            beta0, psi_h = self._choose_yaw(psi)
            self.R_hang = _rz(psi_h)
            r_hand = _down_hand(beta0)
            self.quat = _mat_to_quat(r_hand)
            u = np.array([math.cos(psi), math.sin(psi), 0.0])
            self.g_pos = m_pos - u * (self.r_out - self.wall / 2) + np.array([0, 0, self.half_h - self.grasp_depth])
            self.grip = 0.0
            self._next("approach")
        if self.phase == "approach":
            self.grip = self.open_cmd
            if self._goto(obs, self.g_pos + [0, 0, self.hover], self.free_speed) and self.k > 5:
                self._next("descend")
        elif self.phase == "descend":
            if self._goto(obs, self.g_pos, self.thread_speed * 2):
                self._next("close")
        elif self.phase == "close":
            self.grip = 1.0
            if self.k >= self.grip_steps:
                self._next("lift")
        elif self.phase == "lift":
            if self._goto(obs, self.g_pos + [0, 0, self.lift], self.thread_speed * 2):
                self.rel = []
                self._next("measure")
        elif self.phase == "measure":
            r_e = quat_to_mat(obs.ee_quat)
            self.rel.append(r_e.T @ (m_pos - obs.ee_pos))
            if self.k >= self.measure_steps:
                self.T_rel = (np.mean(self.rel, axis=0), r_e.T @ m_rot)
                self.drift_ref = self.T_rel[0]
                self._next("carry")
        elif self.phase == "carry":
            # up to a height where the mug's bottom clears the peg tip, then over the pre-thread point
            z_safe = self.tip[2] + self.half_h + self.clear_above
            goal = obs.ee_pos.copy()
            goal[2] = max(goal[2], z_safe + (obs.ee_pos[2] - m_pos[2]))
            if self._goto(obs, goal, self.carry_speed):
                self._next("turn")
        elif self.phase == "turn":
            pre = self._mug_at_s(self.pre_s)
            p, q = self._ee_for_mug(pre, self.R_hang)
            above = p.copy()
            above[2] = max(above[2], self.ref[2])
            self.quat = q
            if self._goto(obs, above, self.carry_speed) and self._quat_close(obs.ee_quat, q):
                self._next("pre")
        elif self.phase == "pre":
            p, q = self._ee_for_mug(self._mug_at_s(self.pre_s), self.R_hang)
            if self._goto(obs, p, self.carry_speed / 2):
                self._next("thread")
        elif self.phase == "thread":
            p, _ = self._ee_for_mug(self._mug_at_s(self.thread_s), self.R_hang)
            if self._goto(obs, p, self.thread_speed):
                self._next("seat")
        elif self.phase == "seat":
            # lower until the inside of the handle's top bar touches the top of the peg, then a bit more
            peg_top = self.peg_r / math.cos(math.asin(float(self.axis[2])))
            dz = -(self.loop_top - self.loop_c[2] - peg_top) - self.seat_drop
            p, _ = self._ee_for_mug(self._mug_at_s(self.thread_s, dz=dz), self.R_hang)
            if self._goto(obs, p, self.thread_speed) or self.k > 60:
                self._next("release")
        elif self.phase == "release":
            self.grip = 0.0
            if self.k >= 8:
                self.up = obs.ee_pos + [0, 0, 0.08]
                self._next("rise")
        elif self.phase == "rise":
            if self._goto(obs, self.up, self.thread_speed * 2) or self.k > 80:
                back = obs.ee_pos.copy()
                back[:2] += self.axis[:2] / max(np.linalg.norm(self.axis[:2]), 1e-9) * 0.12
                self.back = back
                self._next("retreat")
        elif self.phase == "retreat":
            if self._goto(obs, self.back, self.free_speed) or self.k > 80:
                self._next("wait")
        elif self.phase == "back_off":
            # the mug slipped: open, rise, and plan again from wherever it is now
            self.grip = 0.0
            up = obs.ee_pos.copy()
            up[2] = max(up[2], 0.25)
            if self.k > 8 and (self._goto(obs, up, self.free_speed) or self.k > 60):
                self.ref = None
                self._next("plan")
                self.ref = obs.ee_pos.copy()
        # "wait" and "give_up": hold still
        return self._act()

    def _track(self, obs: Observation, m_pos: np.ndarray, m_rot: np.ndarray) -> None:
        """Keep the mug-in-hand transform current: the mug turns a little in a rim pinch whenever
        the hand turns or the handle touches something. Rotation from the observed orientation,
        position low-pass filtered (positions are noisy)."""
        r_e = quat_to_mat(obs.ee_quat)
        p_rel = r_e.T @ (m_pos - obs.ee_pos)
        p0, _ = self.T_rel
        self.T_rel = (p0 + self.track_alpha * (p_rel - p0), r_e.T @ m_rot)

    def _slipped(self, obs: Observation, m_pos: np.ndarray) -> bool:
        if not hasattr(self, "drift_ref") or self.phase in ("plan", "approach", "descend", "close", "lift", "measure"):
            return False
        r_e = quat_to_mat(obs.ee_quat)
        return float(np.linalg.norm(r_e.T @ (m_pos - obs.ee_pos) - self.drift_ref)) > self.drift_tol

    @staticmethod
    def _quat_close(a, b, tol: float = 0.05) -> bool:
        return 2 * math.acos(min(1.0, abs(float(np.dot(a, b))))) < tol
