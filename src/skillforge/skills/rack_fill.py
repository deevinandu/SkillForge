"""FillRack v0.1: put several tubes into their rack slots, one after another.

For an `all_of` task whose conditions are each `object_in_region` (tube -> slot), FillRack runs one
`TubeIntoRack` per condition, in the task's order, each on a sub-task whose success is that one
condition. Neighbouring slots are close (25 mm pitch on the default rack), so every TubeIntoRack
closes its fingers across the rack row (`closing_axis`, perpendicular to the row) and opens them
only a little on release (`release_margin`); it also carries over tubes already standing in the
rack. If one tube fails, FillRack stops and holds: the next tube is not attempted.

Loadable as a `SkillSource` (imports only `skillforge.*` and numpy).
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
from skillforge.skills.tube_rack import TubeIntoRack


class FillRack:
    name = "fill_rack"
    version = "0.1.0"

    release_margin = 0.012  # metres the fingers open beyond the tube's diameter on release

    def reset(self, task: TaskSpec) -> None:
        conds = success_conditions(task.success)
        if not isinstance(task.success, AllOf) or not all(isinstance(c, ObjectInRegion) for c in conds):
            raise ValueError("FillRack needs an all_of task of object_in_region conditions")
        self.task = task
        centres = np.array([task.scene.region(c.region).center[:2] for c in conds], dtype=float)
        # the rack row runs along the direction the slot centres spread most; close across it
        row = centres.max(axis=0) - centres.min(axis=0)
        across = (1.0, 0.0, 0.0) if row[1] >= row[0] else (0.0, 1.0, 0.0)
        self.subtasks = [task.model_copy(update={"success": c}) for c in conds]
        self.skills = []
        for sub in self.subtasks:
            sk = TubeIntoRack()
            sk.closing_axis, sk.release_margin = across, self.release_margin
            sk.reset(sub)
            self.skills.append(sk)
        self.i = 0
        self.log: list[str] = []

    @property
    def current(self) -> TubeIntoRack:
        return self.skills[min(self.i, len(self.skills) - 1)]

    def act(self, obs: Observation) -> Action:
        sk = self.current
        status = sk.runner.status
        if not status.running and not status.failed and self.i < len(self.skills) - 1:
            self.log.append(f"{sk.ctx.steps}:done:{sk.task.success.object}")
            self.i += 1
            sk = self.current
        elif status.failed and not self.log[-1:] == [f"failed:{sk.task.success.object}"]:
            self.log.append(f"failed:{sk.task.success.object}")
        return sk.act(obs)
