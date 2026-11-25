"""TubeIntoRack v0.2: stand a test tube upright in a rack slot (a tight insertion).

Built on PickPlace (same planning, monitors and recovery) with the parts a tall, thin tube and a
slot about 2 mm wider than the tube need:
- top-down grasp `grasp_depth` below the tube's top: as deep as the hand allows (the hand body sits
  3.7 cm above the fingertip centre), so the tube does not pivot out of the fingers;
- a short checked lift (`lift_height`), then a vertical rise to a height where the tube's bottom
  clears the rack and every standing tube (`carry_clearance`), then the carry over the slot. Rise,
  carry, alignment and insertion all follow a reference that moves at most `carry_step` /
  `insert_step` per step with its speed ramped by `carry_accel`: a tube held by finger friction
  slips under a sudden start;
- before inserting, the tube is aligned over the slot with its tube-to-hand offset averaged over
  the carry, at a height where its bottom clears the rack (`clear_height`);
- the insertion keeps correcting x/y as it descends. If the tube stops going down while the hand
  does (`jam_tol`: it hit the rack edge and slid up in the fingers), it backs off, realigns and
  tries again (`jam_retries`), without letting go;
- for racks with neighbours: `closing_axis` picks the finger closing direction and
  `release_margin` opens the fingers only a little on release (see `FillRack`).

v0.2 over v0.1: the ramped reference (the old limiter stepped from the measured EE, which lags,
so the arm moved far slower than commanded), the split lift/rise/carry, and the neighbour options.

The rack top is the highest top face among the task's fixed objects. Loadable as a `SkillSource`
(imports only `skillforge.*` and numpy).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Observation, TaskSpec
from skillforge.skills.grasp import WORLD_DOWN, plan_grasps, quat_to_mat, select_grasp
from skillforge.skills.pick_place import PickPlace, _resting_half_height
from skillforge.skills.primitives import (
    Context,
    Do,
    Hold,
    MoveTo,
    Retry,
    Sequence,
    approach,
    descend_until,
    grip,
    hold_check,
    lift,
    release,
    retreat,
)


def _top(o) -> float:
    return float(o.pose.pos[2]) + _resting_half_height(o)


class TubeIntoRack(PickPlace):
    name = "tube_into_rack"
    version = "0.2.0"

    grasp_depth = 0.03  # metres below the tube's top for the fingertip centre
    carry_step = 0.012  # metres per control step while carrying (0.24 m/s at 20 Hz)
    insert_step = 0.002  # metres per control step while inserting (4 cm/s at 20 Hz)
    clear_height = 0.015  # metres between the tube's bottom and the rack top before inserting
    align_tol = 0.0015  # metres: x/y error over the slot that counts as aligned
    align_steps = 6  # steps the alignment must hold before inserting
    jam_tol = 0.006  # metres the tube may slide up in the fingers before it counts as jammed
    jam_retries = 3
    lift_height = 0.03  # metres: the checked lift off the table; the rest of the climb is ramped
    carry_clearance = 0.04  # metres between the held tube's bottom and the tallest thing it passes over
    carry_accel = 0.0003  # metres per step, per step: the carry's speed ramp (0.12 m/s^2 at 20 Hz)
    # Racks with neighbours: close the fingers along this world axis (None = any) and open them
    # only to the tube's diameter + `release_margin` on release (None = fully), so the fingers
    # do not hit tubes already standing in the next slots.
    closing_axis: tuple[float, float, float] | None = None
    release_margin: float | None = None

    def reset(self, task: TaskSpec) -> None:
        fixed = [o for o in task.scene.objects if o.fixed]
        self.rack_top = max((_top(o) for o in fixed), default=0.0)
        self.half_len = _resting_half_height(task.scene.object(task.success.object))
        super().reset(task)  # builds the tree, which reads the two values above

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps
        slot = np.asarray(self.region.center, dtype=float)

        below_tcp = 2 * self.half_len - self.grasp_depth  # tube length hanging below the fingertips

        def safe_z(obs, c):
            # EE height at which the held tube's bottom clears the rack and every tube standing
            # anywhere (`carry_clearance` above the tallest top); at least `hover` above the start.
            tall = max(self.rack_top, self._tallest_other(obs, c))
            return max(c.bb.get("rest_z", self.rest_height) + hv, tall + below_tcp + self.carry_clearance)

        def lift_z(obs, c):
            return c.bb.get("rest_z", self.rest_height) + self.lift_height

        def note_xy(obs, c):
            c.bb["rise_xy"] = obs.ee_pos[:2].copy()
            return None

        def risen(obs, c):
            return np.array([*c.bb["rise_xy"], safe_z(obs, c)])

        def grasp_target(obs, c):
            p = c.obj_pos.copy()
            p[2] = self.rest_height if abs(p[2] - self.rest_height) < 0.025 else max(p[2], self.rest_height)
            return p + quat_to_mat(c.bb["grasp_obj_quat"]) @ c.bb["grasp"]

        def offset(obs, c):
            """Tube centre minus EE, averaged over the carry (the raw view is noisy)."""
            if c.obj in obs.objects and c.monitor.armed:
                c.bb.setdefault("offsets", []).append(np.asarray(obs.objects[c.obj].pos) - obs.ee_pos)
            offs = c.bb.get("offsets") or []
            return c.monitor.offset if len(offs) < 3 else np.mean(offs[-self.offset_window :], axis=0)

        def ee_for(bottom_z: float, o: np.ndarray) -> np.ndarray:
            """EE target that puts the tube's bottom at `bottom_z`, centred over the slot."""
            return np.array([slot[0] - o[0], slot[1] - o[1], bottom_z + self.half_len - o[2]])

        def over_slot(obs, c):
            # align with the bottom `clear_height` above the rack, or above the seated height when
            # the spot is higher than the rack (e.g. on top of other objects in a stack)
            bottom = max(self.rack_top, slot[2] - self.half_len)
            return ee_for(bottom + self.clear_height, offset(obs, c))

        def seated(obs, c):
            return ee_for(slot[2] - self.half_len, offset(obs, c))

        def rate_limited(goal, step, key):
            """A reference point that moves toward `goal` at most `step` per control step, ramping its
            speed up and down by `carry_accel` (a smooth start and stop: a tube held by finger friction
            slips under a sudden start). The reference advances on its own clock rather than from the
            measured EE, which lags its target, so the commanded speed is the speed actually driven."""

            def f(obs, c):
                g = goal(obs, c)
                ref = c.bb.get(key + "_ref")
                if ref is None:
                    ref = obs.ee_pos.copy()
                d = g - ref
                n = float(np.linalg.norm(d))
                a = self.carry_accel
                lim = min(step, c.bb.get(key, 0.0) + a, max(a, float(np.sqrt(2.0 * a * n))))
                c.bb[key] = lim
                ref = g.copy() if n <= lim else ref + d * (lim / n)
                c.bb[key + "_ref"] = ref
                return ref

            return f

        def near(goal, eps):
            return lambda obs, c: float(np.linalg.norm(obs.ee_pos - goal(obs, c))) < eps

        def aligned(obs, c):
            e = obs.ee_pos - over_slot(obs, c)
            c.bb["aligned_n"] = c.bb.get("aligned_n", 0) + 1 if float(np.linalg.norm(e[:2])) < self.align_tol else 0
            return c.bb["aligned_n"] >= self.align_steps

        def jam_check(obs, c):
            held = hold_check("grasp_slip")(obs, c)
            if held:
                return held
            if c.obj in obs.objects and c.bb.get("carry_dz") is not None:
                slid = (obs.objects[c.obj].pos[2] - obs.ee_pos[2]) - c.bb["carry_dz"]
                if slid > self.jam_tol:
                    return f"jam:tube slid {slid * 1000:.0f} mm up in the fingers"
            return None

        def note_carry(obs, c):
            c.bb["carry_dz"] = float(offset(obs, c)[2])
            c.bb["aligned_n"] = 0
            return None

        def above_slot(obs, c):
            p = over_slot(obs, c)
            p[2] = max(p[2], safe_z(obs, c))
            return p

        rise = MoveTo(rate_limited(risen, self.carry_step, "v_rise"), gripper=1.0, tol=1e-9,
                      until=near(risen, 2 * tol), check=hold_check("grasp_slip"), timeout=200, name="rise")
        carry_high = MoveTo(rate_limited(above_slot, self.carry_step, "v_high"), gripper=1.0, tol=1e-9,
                            until=near(above_slot, 2 * tol), check=hold_check("grasp_slip"), timeout=300, name="carry_high")
        carry = MoveTo(rate_limited(over_slot, self.carry_step, "v_low"), gripper=1.0, tol=1e-9,
                       until=near(over_slot, 2 * tol), check=hold_check("grasp_slip"), timeout=300, name="carry")
        insert_once = Sequence(
            [
                MoveTo(rate_limited(over_slot, self.insert_step, "v_align"), gripper=1.0, tol=1e-9, until=aligned,
                       check=hold_check("grasp_slip"), timeout=120, name="align"),
                Do(note_carry, name="note_carry"),
                MoveTo(rate_limited(seated, self.insert_step, "v_insert"), gripper=1.0, tol=1e-9,
                       until=near(seated, 0.002), check=jam_check, timeout=200, name="insert"),
            ],
            name="insert_once",
        )
        insert = Retry(insert_once, {"jam": self.jam_retries}, on_retry=self._on_jam)
        attempt = Sequence(
            [
                Do(self._disarm, name="disarm"),
                retreat(safe_z, tol=tol, name="clear"),
                Do(self._plan, name="plan_grasp"),
                approach(grasp_target, hover=hv, tol=tol),
                descend_until(grasp_target, tol=tol),
                grip(settle=st),
                lift(lift_z, tol=tol),
                Do(note_xy, name="note_xy"),
                rise,
                carry_high,
                carry,
                insert,
                release(settle=st, gripper=self._release_cmd()),
                retreat(safe_z, tol=tol, gripper=self._release_cmd()),
                Do(self._clear_samples, name="clear_samples"),
                Hold(0.0, steps=self.verify_samples, check=self._sample, name="observe"),
                Do(self._verify, name="verify"),
            ],
            name="tube_into_rack",
        )
        budgets = {"grasp_miss": self.miss_retries, "grasp_slip": self.slip_regrasps, "wrong_placement": self.place_retries}
        return Retry(attempt, budgets, on_retry=self._on_retry)

    # -- steps ---------------------------------------------------------------

    def _release_cmd(self) -> float:
        if self.release_margin is None:
            return 0.0
        width = 2 * float(self.spec.size[0]) + self.release_margin
        return float(np.clip(1.0 - width / self.max_width, 0.0, 0.45))  # > 0.5 reads as "closed"

    def _tallest_other(self, obs: Observation, c: Context) -> float:
        """Highest observed top among the other movable objects (e.g. tubes already racked)."""
        tops = [0.0]
        for o in self.task.scene.objects:
            if o.fixed or o.name == c.obj or o.name not in obs.objects:
                continue
            tops.append(float(obs.objects[o.name].pos[2]) + _resting_half_height(o))
        return max(tops)

    def _on_jam(self, c: Context, reason: str) -> None:
        c.bb["aligned_n"] = 0  # back off: the align step starts from wherever the hand is now
        for k in ("v_align", "v_insert"):
            c.bb[k], c.bb[k + "_ref"] = 0.0, None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        """Top-down grasps only (the tube must go into the slot vertically), `grasp_depth` deep."""
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(c.obj_pos[2])
        c.bb["offsets"] = []
        for k in ("v_rise", "v_high", "v_low", "v_align", "v_insert"):  # every ramp starts from rest
            c.bb[k], c.bb[k + "_ref"] = 0.0, None
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        r = quat_to_mat(q_obj)
        if abs(float(r[2, 2])) < 0.95:  # the tube's axis (object z) must be near vertical
            return "grasp_miss:tube is not upright"
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj, depth=self.grasp_depth, max_tilt=0.3)
        # end-cap grasps only: approach along the tube's axis, from above
        grasps = [g for g in grasps if abs(g.approach[2]) > 0.99 and float(np.dot(r @ g.approach, WORLD_DOWN)) > 0.95]
        if self.closing_axis is not None:
            ax = np.asarray(self.closing_axis, dtype=float)
            grasps = [g for g in grasps if abs(float(np.dot(r @ quat_to_mat(g.quat)[:, 1], ax))) > 0.9] or grasps
        if not grasps:
            return "grasp_miss:no top-down grasp on the tube"
        _, quat, g = select_grasp(grasps, c.obj_pos, q_obj, ee_quat=obs.ee_quat, rank=c.bb["rank"])
        c.bb["grasp"], c.bb["grasp_obj_quat"] = g.pos, q_obj
        c.quat = quat
        return None
