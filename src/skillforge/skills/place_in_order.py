"""PlaceInOrder v0.1: place several objects at exact spots, one after another (e.g. a block pyramid).

For an `all_of` task whose conditions are each `object_in_region`, where each region is a tight
spot marking exactly where the object must end up (its centre is the object's resting centre, as
in a stack), PlaceInOrder runs one PickPlace per condition in the task's order. Order matters for
a stack: the task lists the bottom row first.

Each PickPlace:
- aims at the spot's centre, not "region floor + half height" (PickPlace's default suits bins,
  and here would release the object above its spot);
- closes its fingers across the row the spots form, so the fingers stay clear of neighbours.

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
from skillforge.skills.pick_place import PickPlace


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

    def reset(self, task: TaskSpec) -> None:
        super().reset(task)
        self.place_obj = np.asarray(self.region.center, dtype=float).copy()


class PlaceInOrder:
    name = "place_in_order"
    version = "0.1.0"

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
            sk = SpotPlace()
            sk.closing_axis = across
            sk.reset(task.model_copy(update={"success": c}))
            self.skills.append(sk)
        self.i = 0

    def act(self, obs: Observation) -> Action:
        sk = self.skills[self.i]
        st = sk.runner.status
        if not st.running and not st.failed and self.i < len(self.skills) - 1:
            self.i += 1
            sk = self.skills[self.i]
        return sk.act(obs)
