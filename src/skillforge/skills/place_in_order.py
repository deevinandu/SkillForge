"""PlaceInOrder v0.2: place several objects at exact spots, one after another (e.g. a block pyramid).

For an `all_of` task whose conditions are each `object_in_region`, where each region is a tight
spot marking exactly where the object must end up (its centre is the object's resting centre, as
in a stack), PlaceInOrder runs one PickPlace per condition in the task's order. Order matters for
a stack: the task lists the bottom row first.

Boxes (and anything that is not a cylinder) are placed with SpotPlace, a PickPlace that:
- aims at the spot's centre, not "region floor + half height" (PickPlace's default suits bins,
  and here would release the object above its spot);
- closes its fingers across the row the spots form, so the fingers stay clear of neighbours.

Cylinders standing upright (v0.2) are placed with TubeIntoRack instead (see `upright_accel`).

If one object fails, PlaceInOrder stops and holds. Loadable as a `SkillSource` (imports only
`skillforge.*` and numpy).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import (
    Action,
    AllOf,
    ObjectInRegion,
    Observation,
    TaskSpec,
    success_conditions,
)
from skillforge.skills.pick_place import PickPlace, _resting_half_height
from skillforge.skills.tube_rack import TubeIntoRack


class SpotPlace(PickPlace):
    """PickPlace that puts the object's centre at its region's centre (an exact spot)."""

    name = "spot_place"
    version = "0.1.0"

    # PickPlace's 6 mm "arrived" tolerance suits a bin; here it released every block 5-11 mm short
    # of its spot, on the side it was carried in from. 2 mm is close enough for a stack.
    tol = 0.002
    # Closing across the row, the block is held along the row by friction alone and slides in the
    # fingers on the carry (4-15 mm, measured). A 40-sample average of the in-hand offset lags that
    # by ~5 mm; the last 8 samples track it.
    offset_window = 8

    carry_clearance = 0.04  # metres between a held object's bottom and the tops of standing ones

    def reset(self, task: TaskSpec) -> None:
        # A tall object held near its top hangs well below the fingers: carry it high enough that
        # its bottom clears neighbours as tall as itself (PickPlace's 10 cm hover suits a 4 cm
        # block, but swept an 8.4 cm cylinder through its neighbours).
        spec = task.scene.object(task.success.object)
        self.hover = max(type(self).hover, 2 * _resting_half_height(spec) + self.carry_clearance)
        super().reset(task)
        self.place_obj = np.asarray(self.region.center, dtype=float).copy()


class PlaceInOrder:
    name = "place_in_order"
    version = "0.2.0"

    # Upright cylinders (tubes, cans) are placed with TubeIntoRack: a speed-ramped carry, alignment
    # above the spot and a slow, corrected descent. Gripped across the body, a cylinder touches each
    # pad along a line and swings about it under acceleration, so it needs a gentle ramp: for 2 in x
    # 3.3 in, 100 g cylinders the pyramid went 75 -> 96 -> 99/100 at 0.0003 -> 0.0002 -> 0.00012.
    upright_accel = 0.00012  # metres per step, per step (0.048 m/s^2 at 20 Hz)
    release_margin = 0.012  # metres the fingers open beyond the cylinder's diameter on release

    def reset(self, task: TaskSpec) -> None:
        conds = success_conditions(task.success)
        if not isinstance(task.success, AllOf) or not all(isinstance(c, ObjectInRegion) for c in conds):
            raise ValueError("PlaceInOrder needs an all_of task of object_in_region conditions")
        self.task = task
        centres = np.array([task.scene.region(c.region).center[:2] for c in conds], dtype=float)
        row = centres.max(axis=0) - centres.min(axis=0)
        across = (1.0, 0.0, 0.0) if row[1] >= row[0] else (0.0, 1.0, 0.0)
        self.skills = []
        for c in conds:
            sk = self._placer(task.scene.object(c.object).shape)
            sk.closing_axis = across
            sk.reset(task.model_copy(update={"success": c}))
            self.skills.append(sk)
        self.i = 0

    def _placer(self, shape: str):
        if shape == "cylinder":
            sk = TubeIntoRack()
            sk.carry_accel, sk.release_margin = self.upright_accel, self.release_margin
            return sk
        return SpotPlace()

    def act(self, obs: Observation) -> Action:
        sk = self.skills[self.i]
        st = sk.runner.status
        if not st.running and not st.failed and self.i < len(self.skills) - 1:
            self.i += 1
            sk = self.skills[self.i]
        return sk.act(obs)
