"""SortTubes v0.1: sort tubes into racks by cap colour, deciding each colour by LOOKING.

For a task whose success is `color_in_region` conditions (one rack region per colour, each region
spanning that rack's slots) and whose scene lists the wrist camera in `obs_cameras`:

1. Look. The arm holds the wrist camera straight above each tube in turn and reads the colour
   of the tube's top from the centre of the image (`look_height` above the top, mean RGB of a
   `patch` x `patch` window averaged over `look_frames` frames). The colour is the nearest
   reference colour in chromaticity (r, g, b divided by their sum, so lighting matters less)
   among the colours the task sorts. The object names and the spec's colours are NOT used to
   decide: with `shuffle_colors` they do not say which tube is which colour.
2. Sort. Each tube goes into the next free slot of its colour's rack, through `TubeIntoRack`
   with the fingers closing across the rack row and a narrow release (as in `FillRack`).

Slots are spread evenly along each region's long axis, `slot_pitch` apart. The wrist camera's
pose relative to the fingertip centre (`cam_offset`, hand frame) is a calibration constant:
on hardware it comes from the hand-eye calibration.

If a colour cannot be read (no tube in the patch), the tube is skipped and logged. Loadable as a
`SkillSource` (imports only `skillforge.*` and numpy).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import (
    Action,
    ColorInRegion,
    ObjectInRegion,
    Observation,
    Region,
    TaskSpec,
    success_conditions,
)
from skillforge.skills.grasp import DOWN, quat_to_mat
from skillforge.skills.pick_place import _resting_half_height
from skillforge.skills.primitives import Context, Do, Hold, MoveTo, Runner, Sequence
from skillforge.skills.tube_rack import TubeIntoRack

# How each colour looks to the wrist camera at the look pose (RGB, 0..1). Red and blue were measured
# in the sim (the top faces render washed out under the overhead light); the rest are nominal. On
# hardware these come from a colour calibration under the cell's lighting.
REFERENCE_RGB = {
    "red": (1.0, 0.63, 0.63), "blue": (0.63, 0.94, 1.0), "green": (0.2, 0.7, 0.3),
    "yellow": (0.95, 0.75, 0.1), "orange": (0.95, 0.5, 0.1), "purple": (0.55, 0.25, 0.75),
    "white": (0.92, 0.92, 0.92), "grey": (0.55, 0.55, 0.55), "black": (0.08, 0.08, 0.08),
}
# What the camera sees when no tube is under it (the table, measured at three spots): a match
# here means "nothing to read", never a colour.
BACKGROUND_RGB = [(1.0, 1.0, 0.85), (1.0, 0.84, 0.65), (1.0, 1.0, 0.92)]


def chroma(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return rgb / max(float(rgb.sum()), 1e-6)


def classify(rgb: np.ndarray, candidates: list[str], max_dist: float = 0.12) -> str | None:
    """Nearest candidate colour by chromaticity; None if the background is nearer, or nothing is
    within `max_dist`."""
    c = chroma(rgb)
    d = {name: float(np.linalg.norm(c - chroma(REFERENCE_RGB[name]))) for name in candidates}
    best = min(d, key=d.get)
    bg = min(float(np.linalg.norm(c - chroma(b))) for b in BACKGROUND_RGB)
    return best if d[best] <= max_dist and d[best] < bg else None


class SortTubes:
    name = "sort_tubes"
    version = "0.1.0"

    camera = "wrist"
    cam_offset = (0.07, 0.0, -0.0634)  # wrist camera minus fingertip centre, hand frame (metres)
    look_height = 0.12  # metres from the camera to the tube's top
    patch = 9  # pixels: side of the centre window whose mean colour is read
    look_frames = 3
    settle_steps = 6
    slot_pitch = 0.025
    release_margin = 0.012

    def reset(self, task: TaskSpec) -> None:
        conds = success_conditions(task.success)
        if not conds or not all(isinstance(c, ColorInRegion) for c in conds):
            raise ValueError("SortTubes needs color_in_region conditions")
        if self.camera not in task.scene.obs_cameras:
            raise ValueError(f"SortTubes needs the '{self.camera}' camera in scene.obs_cameras")
        self.task = task
        self.rack_of = {c.color: task.scene.region(c.region) for c in conds}
        self.free = {col: self._slots(r) for col, r in self.rack_of.items()}
        self.tubes = [o for o in task.scene.objects if not o.fixed and o.color]
        self.seen: dict[str, str | None] = {}
        self.log: list[str] = []
        self.ctx = Context(obj=None)
        self.ctx.quat = DOWN.copy()
        self.looker = Runner(Sequence([self._look(o) for o in self.tubes], name="look"), self.ctx)
        self.queue: list[TubeIntoRack] = []
        self.i = -1  # -1: still looking

    # -- look ------------------------------------------------------------------

    def _look_pos(self, obs: Observation, name: str) -> np.ndarray:
        o = self.task.scene.object(name)
        p = np.asarray(obs.objects[name].pos, dtype=float)
        top = p[2] + _resting_half_height(o)
        cam = quat_to_mat(self.ctx.quat) @ np.asarray(self.cam_offset)  # camera minus TCP, world
        return np.array([p[0] - cam[0], p[1] - cam[1], top + self.look_height - cam[2]])

    def _look(self, o):
        name = o.name

        def target(obs, c):
            if f"look:{name}" not in c.bb:  # aim once, from the first view: then hold still
                c.bb[f"look:{name}"] = self._look_pos(obs, name)
            return c.bb[f"look:{name}"]

        def sample(obs, c):
            img = obs.images.get(self.camera)
            if img is not None:
                h, w = img.shape[:2]
                k = self.patch // 2
                win = img[h // 2 - k : h // 2 + k + 1, w // 2 - k : w // 2 + k + 1, :3]
                c.bb.setdefault(f"rgb:{name}", []).append(win.reshape(-1, 3).mean(axis=0) / 255.0)
            return None

        def decide(obs, c):
            rgbs = c.bb.get(f"rgb:{name}", [])[-self.look_frames :]
            col = classify(np.mean(rgbs, axis=0), list(self.rack_of)) if rgbs else None
            self.seen[name] = col
            self.log.append(f"{c.steps}:saw:{name}:{col}")
            return None

        return Sequence(
            [
                MoveTo(target, gripper=0.0, tol=0.004, timeout=150, name=f"look_at_{name}"),
                Hold(0.0, steps=self.settle_steps, name="settle"),
                Hold(0.0, steps=self.look_frames, check=sample, name="sample"),
                Do(decide, name="decide"),
            ],
            name=f"look_{name}",
        )

    # -- sort ------------------------------------------------------------------

    def _slots(self, r: Region) -> list[np.ndarray]:
        """Slot centres spread evenly along the region's long horizontal axis."""
        c, h = np.asarray(r.center, dtype=float), np.asarray(r.half_extents, dtype=float)
        ax = int(np.argmax(h[:2]))
        n = max(1, int(round((2 * h[ax] - 2 * h[1 - ax]) / self.slot_pitch)) + 1)
        span = (n - 1) * self.slot_pitch
        out = []
        for k in range(n):
            p = c.copy()
            p[ax] += -span / 2 + k * self.slot_pitch
            out.append(p)
        return out

    def _plan_sort(self) -> None:
        for o in self.tubes:
            col = self.seen.get(o.name)
            if col is None or not self.free.get(col):
                self.log.append(f"skip:{o.name}:{'unread' if col is None else 'rack full'}")
                continue
            slot = self.free[col].pop(0)
            rname = f"_slot_for_{o.name}"
            reg = Region(name=rname, center=tuple(slot), half_extents=(0.004, 0.004, 0.004))
            scene = self.task.scene.model_copy(update={"regions": (*self.task.scene.regions, reg)})
            sub = self.task.model_copy(update={"scene": scene, "success": ObjectInRegion(object=o.name, region=rname)})
            sk = TubeIntoRack()
            row = self.rack_of[col].half_extents
            sk.closing_axis = (1.0, 0.0, 0.0) if row[1] >= row[0] else (0.0, 1.0, 0.0)
            sk.release_margin = self.release_margin
            sk.reset(sub)
            self.queue.append(sk)
            self.log.append(f"plan:{o.name}->{col}")

    def act(self, obs: Observation) -> Action:
        if self.i < 0:
            st = self.looker.status
            if st.running:
                return self.looker.act(obs)
            self._plan_sort()
            self.i = 0
        while self.i < len(self.queue) - 1:
            st = self.queue[self.i].runner.status
            if st.running or st.failed:
                break
            self.i += 1
        if not self.queue:
            return Action(ee_pos=obs.ee_pos, ee_quat=self.ctx.quat, gripper=0.0)
        return self.queue[self.i].act(obs)
