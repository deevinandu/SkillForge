"""PickPlace v0.2: scripted closed-loop pick-and-place built from `primitives`, with recovery.

Over v0.1 (still available as `skillforge.skills.pick_place_v01.PickPlaceV01`):
- grasp orientation from the grasp planner (`grasp.py`) and the observed object orientation;
- filtered object observations instead of chasing raw noisy positions;
- a missed grasp (object not following the gripper on lift, or fingers closed on nothing) is
  retried with the next-ranked grasp (`miss_retries`, default once);
- a slip while carrying (object leaves the gripper, or the measured width collapses) triggers a
  re-grasp from wherever the object ended up (`slip_regrasps`, default twice);
- after release, if the object is observed outside the goal region it is picked again once.

Loadable as a `SkillSource`: one file, imports only `skillforge.*` and numpy (absolute imports).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, ObjectSpec, Observation, TaskSpec
from skillforge.skills.grasp import (
    DOWN,
    plan_grasps,
    quat_angle,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    select_grasp,
    yaw_of,
)
from skillforge.skills.primitives import (
    Context,
    Do,
    Retry,
    Runner,
    Sequence,
    approach,
    descend_until,
    grip,
    lift,
    place,
    release,
    retreat,
    transport,
)


def _resting_half_height(o: ObjectSpec) -> float:
    if o.shape == "box":
        return float(o.size[2])
    if o.shape == "cylinder":
        return float(o.size[1])
    if o.shape == "sphere":
        return float(o.size[0])
    return 0.0


class _RobustContext(Context):
    """Context that also keeps a median estimate of the object's position while it rests.

    The base filter snaps to the raw reading on jumps > 2 cm, which observation noise alone
    triggers often, so grasp targets inherited the full noise. The median of recent raw readings
    is used for grasp targets instead; the buffer restarts when the object clearly moved.
    """

    buf_len = 25
    moved = 0.04
    moved_steps = 3

    def update(self, obs: Observation) -> None:
        super().update(obs)
        if self.obj is None or self.obj not in obs.objects:
            return
        raw = np.asarray(obs.objects[self.obj].pos, dtype=float).copy()
        buf = self.bb.setdefault("buf", [])
        far = self.bb.setdefault("far", [])
        if buf and np.linalg.norm(raw - np.median(np.array(buf), axis=0)) > self.moved:
            far.append(raw)
            if len(far) >= self.moved_steps:
                buf[:] = list(far)
                far.clear()
            return
        far.clear()
        buf.append(raw)
        if len(buf) > self.buf_len:
            del buf[0]

    @property
    def rest_pos(self) -> np.ndarray | None:
        buf = self.bb.get("buf")
        if not buf:
            return self.obj_pos
        return np.median(np.array(buf), axis=0)


class PickPlace:
    name = "pick_place"
    version = "0.1.1"

    hover = 0.10  # metres above grasp/place targets for pre-grasp and transport
    tol = 0.006  # metres: position error that counts as "arrived"
    settle_steps = 8  # steps to hold a gripper command before moving on
    miss_retries = 1
    slip_regrasps = 2
    place_retries = 1
    grasp_depth = 0.035# metres past the top surface for the TCP (clipped at the centre)
    min_tcp_z = 0.028  # metres: lowest TCP height, so the fingertips stay clear of the table
    carry_speed = 0.18  # m/s: commanded horizontal speed limit while the gripper is closed
    carry_speed_z = 0.07 # m/s: commanded vertical speed limit while the gripper is closed
    carry_accel = 0.3  # m/s^2 (horizontal; scaled down with the vertical speed limit): commanded acceleration limit while the gripper is closed
    place_lift = 0.012  # metres above resting height at which the object is released
    verify_margin = 0.01  # metres: slack on the region check, which uses noisy observations

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PickPlace needs an object_in_region task")
        self.task = task
        self.spec = task.scene.object(task.success.object)
        self.region = task.scene.region(task.success.region)
        self.max_width = task.scene.robot.max_gripper_width
        center = np.asarray(self.region.center, dtype=float)
        bottom = center[2] - self.region.half_extents[2]
        self.place_obj = np.array([center[0], center[1], max(center[2], bottom + _resting_half_height(self.spec))])
        self.ctx = _RobustContext(obj=self.spec.name)
        # Observation noise reaches 12 mm per axis; the default 3 cm drift threshold then fires on
        # a firmly held object and the retry opens the gripper. A real drop moves far more.
        self.ctx.monitor.dev_tol, self.ctx.monitor.dev_hard, self.ctx.monitor.sustain = 0.045, 0.07, 4
        self.ctx.bb["rank"] = 0
        self.runner = Runner(self._build(), self.ctx)
        self._cmd: np.ndarray | None = None
        self._speed = 0.0
        self._dt = 1.0 / float(task.control_hz)

    def act(self, obs: Observation) -> Action:
        a = self.runner.act(obs)
        if a.mode != "ee" or a.gripper < 0.5:
            self._cmd, self._speed = None, 0.0
            return a
        # Holding: ramp the commanded position (bounded speed and acceleration) so inertial
        # loads on a low-friction / heavy object stay below what the grip can hold.
        if self._cmd is None:
            self._cmd = np.asarray(obs.ee_pos, dtype=float).copy()
        delta = np.asarray(a.ee_pos, dtype=float) - self._cmd
        dist = float(np.linalg.norm(delta))
        u = delta / dist if dist > 1e-9 else np.zeros(3)
        vlim = 1.0 / np.sqrt((u[0] ** 2 + u[1] ** 2) / self.carry_speed**2 + u[2] ** 2 / self.carry_speed_z**2 + 1e-12)
        alim = vlim * self.carry_accel / self.carry_speed
        v = min(vlim, self._speed + alim * self._dt, np.sqrt(2 * alim * dist))
        self._speed = v
        step = v * self._dt
        self._cmd = self._cmd + (delta if dist <= step else delta / dist * step)
        return Action(ee_pos=self._cmd.copy(), ee_quat=a.ee_quat, gripper=a.gripper)

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps

        def safe_z(obs, c):
            # Object's resting height as seen at planning time (it rises with the gripper later).
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + hv

        def grasp_target(obs, c):
            g, q_obj = c.bb["grasp"], c.bb["grasp_obj_quat"]
            p = c.rest_pos + quat_to_mat(q_obj) @ g
            p[2] = max(p[2], self.min_tcp_z)  # keep the fingertips off the table
            return p

        def place_target(obs, c):
            # Stop a little above the floor: the offset is estimated from noisy observations and
            # pushing the held can into the table (more so with action latency) levers it out.
            return self.place_obj - c.monitor.offset + [0.0, 0.0, self.place_lift]

        def above_place(obs, c):
            return place_target(obs, c) + [0.0, 0.0, hv]

        attempt = Sequence(
            [
                Do(self._disarm, name="disarm"),
                retreat(safe_z, tol=tol, name="clear"),
                Do(self._plan, name="plan_grasp"),
                approach(grasp_target, hover=hv, tol=tol),
                descend_until(grasp_target, tol=tol),
                grip(settle=st),
                lift(safe_z, tol=tol),
                transport(above_place, tol=tol),
                place(place_target, tol=tol),
                release(settle=st),
                retreat(safe_z, tol=tol),
                Do(self._verify, name="verify"),
            ],
            name="pick_place",
        )
        budgets = {"grasp_miss": self.miss_retries, "grasp_slip": self.slip_regrasps, "wrong_placement": self.place_retries}
        return Retry(attempt, budgets, on_retry=self._on_retry)

    # -- steps ---------------------------------------------------------------

    def _disarm(self, obs: Observation, c: Context) -> str | None:
        c.monitor.disarm()
        c.bb.pop("rest_z", None)  # re-read from wherever the object is now
        return None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(c.rest_pos[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        # Grasp deeper (near the centre of mass) so the tall object cannot pivot out of the pads.
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj, depth=self.grasp_depth)
        if grasps:
            _, quat, g = select_grasp(grasps, c.obj_pos, q_obj, ee_quat=obs.ee_quat, rank=c.bb["rank"])
            if c.bb["rank"] == 0:
                # Close across the carry direction: measured on dev seeds, inertial load along
                # the closing axis (one pad pushing, the other unloading) pulls the can out of
                # the pads far more often than load across it. Ties: least wrist rotation.
                d = self.place_obj[:2] - c.rest_pos[:2]
                if np.linalg.norm(d) > 1e-6:
                    d = np.append(d / np.linalg.norm(d), 0.0)
                    top = [h for h in grasps if h.score >= grasps[0].score - 0.05]

                    def key(h):
                        qw = quat_mul(q_obj, h.quat)
                        return (round(abs(float(quat_to_mat(qw)[:, 1] @ d)), 2), quat_angle(qw, obs.ee_quat))

                    g = min(top, key=key)
                    quat = quat_mul(q_obj, g.quat)
            c.bb["grasp"], c.bb["grasp_obj_quat"] = g.pos, q_obj
            c.quat = quat
            return None
        if self.spec.shape != "mesh":
            return f"other:no_feasible_grasp(max_width={self.max_width})"
        # Mesh: no planner yet. Top-down at the observed centre, yawed with the object.
        c.bb["grasp"], c.bb["grasp_obj_quat"] = np.zeros(3), np.array([1.0, 0, 0, 0])
        c.quat = quat_mul(quat_from_axis_angle([0, 0, 1], yaw_of(q_obj)), DOWN)
        return None

    def _verify(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return None
        d = np.abs(c.obj_pos - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(c.obj_pos, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
