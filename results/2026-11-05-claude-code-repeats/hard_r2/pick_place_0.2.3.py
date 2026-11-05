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
    Context,
    Do,
    Hold,
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


class PickPlace:
    name = "pick_place"
    version = "0.2.1"

    hover = 0.10  # metres above grasp/place targets for pre-grasp and transport
    tol = 0.006  # metres: position error that counts as "arrived"
    settle_steps = 4  # steps to hold a gripper command before moving on
    miss_retries = 1
    slip_regrasps = 2
    place_retries = 1
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
        self.rest_h = _resting_half_height(self.spec)
        self.ctx = Context(obj=self.spec.name)
        self.ctx.bb["rank"] = 0
        # Running means of the raw (noisy) observations: object position while it sits still,
        # and object-to-EE offset while it is carried.
        self._still_mean, self._still_n = None, 0
        self._off_mean, self._off_n = None, 0
        base_update = self.ctx.update
        self.ctx.update = lambda obs: self._update(obs, base_update)
        self.runner = Runner(self._build(), self.ctx)

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    still_jump = 0.05  # metres: raw deviation from the running mean that means "the object moved"
    still_cap = 40  # samples: cap so the mean can follow slow drift

    def _update(self, obs: Observation, base_update) -> None:
        base_update(obs)
        c = self.ctx
        if c.obj not in obs.objects:
            return
        raw = np.asarray(obs.objects[c.obj].pos, dtype=float)
        if c.monitor.armed:
            # Carried: average the observed offset; the still-object estimate is stale.
            self._still_mean, self._still_n = None, 0
            off = raw - obs.ee_pos
            if self._off_mean is None:
                self._off_mean, self._off_n = off.copy(), 1
            else:
                self._off_n = min(self._off_n + 1, self.still_cap)
                self._off_mean += (off - self._off_mean) / self._off_n
            return
        self._off_mean, self._off_n = None, 0
        if self._still_mean is None or np.linalg.norm(raw - self._still_mean) > self.still_jump:
            self._still_mean, self._still_n = raw.copy(), 1
        else:
            self._still_n = min(self._still_n + 1, self.still_cap)
            self._still_mean += (raw - self._still_mean) / self._still_n
        if self._still_n >= 3:
            est = self._still_mean.copy()
            # Resting on the table (or the bin floor, same height): its centre height is known.
            if abs(est[2] - self.rest_h) < 0.03:
                est[2] = self.rest_h
            c.obj_pos = est
        elif c.obj_pos is not None:
            c.obj_pos = c.obj_pos.copy()
            c.obj_pos[2] = max(c.obj_pos[2], self.rest_h)

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps

        def safe_z(obs, c):
            # Object's resting height as seen at planning time (it rises with the gripper later).
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + hv

        def grasp_target(obs, c):
            g, q_obj = c.bb["grasp"], c.bb["grasp_obj_quat"]
            return c.obj_pos + quat_to_mat(q_obj) @ g

        def place_target(obs, c):
            off = c.monitor.offset
            if self._off_mean is not None and self._off_n >= 3:
                off = self._off_mean
            return self.place_obj - off

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
                Hold(0.0, steps=6, name="observe"),
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
        c.bb["rest_z"] = float(c.obj_pos[2])
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
        d = np.abs(c.obj_pos - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(c.obj_pos, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
