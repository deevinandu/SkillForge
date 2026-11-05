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
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    select_grasp,
    yaw_of,
)
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    HoldMonitor,
    MoveTo,
    Retry,
    Runner,
    Sequence,
    approach,
    descend_until,
    fail,
    grip,
    hold_check,
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


class _RobustMonitor(HoldMonitor):
    """HoldMonitor whose drift test uses the mean of the last few raw offsets instead of a single
    noisy sample: with ~1.5 cm observation noise, single samples cross the 5 cm 'lost' threshold
    by chance every few dozen steps, which aborts good carries and makes the retry drop the cube."""

    window = 8

    def arm(self, obs: Observation, obj_pos: np.ndarray) -> None:
        super().arm(obs, obj_pos)
        self._devs: list[np.ndarray] = []

    def check(self, obs: Observation, obj: str, obj_pos: np.ndarray | None = None) -> str | None:
        if not self.armed:
            return None
        z = float((obs.objects[obj].pos if obj_pos is None else obj_pos)[2])
        self.rise = max(self.rise, z - self.z0)
        if self.w0 > self.width_eps and obs.gripper_width < self.width_eps:
            return "width_collapsed"
        devs = getattr(self, "_devs", [])
        devs.append(np.asarray(obs.objects[obj].pos, dtype=float) - obs.ee_pos - self.offset)
        self._devs = devs[-self.window :]
        if len(self._devs) < 3:
            return None
        dev = float(np.linalg.norm(np.mean(self._devs, axis=0)))
        if dev > self.dev_hard:
            return f"object_left_gripper({dev:.3f}m)"
        self._bad = self._bad + 1 if dev > self.dev_tol else 0
        if self._bad >= self.sustain:
            return f"object_drifting({dev:.3f}m)"
        return None


class _Align(MoveTo):
    """Hold position on `target` until the horizontal error is below `xy_tol` for `stable` steps
    (or `timeout` steps pass; then it finishes anyway rather than failing the attempt)."""

    def __init__(self, target, xy_tol: float = 0.0025, z_tol: float = 0.01, stable: int = 2, **kw):
        super().__init__(target, **kw)
        self.xy_tol, self.z_tol, self.stable = xy_tol, z_tol, stable
        self.ok = 0

    def reset(self) -> None:
        super().reset()
        self.ok = 0

    def tick(self, obs, ctx):
        self.n += 1
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        tgt = np.asarray(self.target(obs, ctx), dtype=float)
        err = obs.ee_pos - tgt
        good = float(np.linalg.norm(err[:2])) < self.xy_tol and abs(float(err[2])) < self.z_tol
        self.ok = self.ok + 1 if good else 0
        if self.ok >= self.stable or self.n > self.timeout:
            return None, DONE
        return Action(ee_pos=tgt, ee_quat=ctx.quat, gripper=self.gripper), RUNNING


class PickPlace:
    name = "pick_place"
    version = "0.2.1"

    hover = 0.10  # metres above grasp/place targets for pre-grasp and transport
    tol = 0.006  # metres: position error that counts as "arrived"
    settle_steps = 10  # steps to hold a gripper command: must exceed the action latency (up to 6)
    miss_retries = 1
    slip_regrasps = 2
    place_retries = 1
    finger_clear = 0.012  # metres: minimum TCP height above the table when grasping
    place_drop = 0.004  # metres: release this far above resting contact
    verify_margin = 0.005  # metres: slack on the region check (on median-filtered observations)

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
        half = _resting_half_height(self.spec)
        self.table_z = float(self.spec.pose.pos[2]) - half
        self.rest_center_z = self.table_z + half
        self.place_obj[2] = max(self.place_obj[2], self.rest_center_z)
        self._free: list[np.ndarray] = []  # raw object positions while not held (median filter)
        self._carry: list[np.ndarray] = []  # raw object-minus-EE offsets while held
        self._grip_ee_z: float | None = None
        self._grip_rest_z = self.rest_center_z
        self.ctx = Context(obj=self.spec.name, monitor=_RobustMonitor())
        self.ctx.bb["rank"] = 0
        self.runner = Runner(self._build(), self.ctx)

    def act(self, obs: Observation) -> Action:
        o = obs.objects.get(self.spec.name)
        if o is not None:
            raw = np.asarray(o.pos, dtype=float)
            if self.ctx.monitor.armed:
                self._free.clear()
                self._carry.append(raw - obs.ee_pos)
                self._carry = self._carry[-40:]
            else:
                if self._free and np.linalg.norm(raw - self._est_free()) > 0.06:
                    self._free.clear()  # object really moved (pushed/dropped): restart
                self._free.append(raw)
                self._free = self._free[-15:]
        return self.runner.act(obs)

    # -- estimates -----------------------------------------------------------

    def _est_free(self) -> np.ndarray:
        return np.median(np.array(self._free), axis=0)

    def _obj_est(self, c: Context) -> np.ndarray:
        """Denoised position of the (not held) object; height snapped to the table if resting."""
        p = self._est_free() if len(self._free) >= 3 else np.array(c.obj_pos, dtype=float)
        if p[2] < self.rest_center_z + 0.025:
            p = p.copy()
            p[2] = self.rest_center_z
        return p

    def _carry_offset(self, c: Context) -> np.ndarray:
        """Object-minus-EE offset while held: median of raw samples; z from grasp geometry."""
        if len(self._carry) >= 3:
            off = np.median(np.array(self._carry), axis=0)
        else:
            off = np.array(c.monitor.offset, dtype=float)
        if self._grip_ee_z is not None:
            off[2] = self._grip_rest_z - self._grip_ee_z
        return off

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps

        def safe_z(obs, c):
            # Object's resting height as seen at planning time (it rises with the gripper later).
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + hv

        def grasp_target(obs, c):
            g, q_obj = c.bb["grasp"], c.bb["grasp_obj_quat"]
            t = self._obj_est(c) + quat_to_mat(q_obj) @ g
            t[2] = max(t[2], self.table_z + self.finger_clear)
            return t

        def place_target(obs, c):
            t = self.place_obj - self._carry_offset(c)
            t[2] += self.place_drop
            return t

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
                Do(self._record_grip, name="record_grip"),
                lift(safe_z, tol=tol),
                transport(above_place, tol=tol),
                _Align(above_place, gripper=1.0, timeout=25, check=hold_check("grasp_slip"), name="align_place"),
                place(place_target, tol=tol),
                _Align(place_target, gripper=1.0, timeout=12, z_tol=0.05, check=hold_check("grasp_slip"), name="settle_place"),
                release(settle=st),
                retreat(safe_z, tol=tol),
                Do(self._verify, name="verify"),
            ],
            name="pick_place",
        )
        budgets = {"grasp_miss": self.miss_retries, "grasp_slip": self.slip_regrasps, "wrong_placement": self.place_retries}
        return Retry(attempt, budgets, on_retry=self._on_retry)

    # -- steps ---------------------------------------------------------------

    def _record_grip(self, obs: Observation, c: Context) -> str | None:
        self._grip_ee_z = float(obs.ee_pos[2])
        self._grip_rest_z = float(c.bb.get("rest_z", self.rest_center_z))
        self._carry.clear()
        return None

    def _disarm(self, obs: Observation, c: Context) -> str | None:
        c.monitor.disarm()
        self._grip_ee_z = None
        c.bb.pop("rest_z", None)  # re-read from wherever the object is now
        return None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(self._obj_est(c)[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj)
        if grasps:
            _, quat, g = select_grasp(grasps, c.obj_pos, q_obj, ee_quat=obs.ee_quat, rank=c.bb["rank"])
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
        p = self._est_free() if len(self._free) >= 5 else c.obj_pos
        d = np.abs(p - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(c.obj_pos, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
