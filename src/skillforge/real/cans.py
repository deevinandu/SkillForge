"""T7: pose estimation for several IDENTICAL upright cylinders (the lab's cans), on a real camera.

T6's `DepthEstimator` tells objects apart by colour (one hue per object name). The lab's cans
all look the same (red label, steel top), so that cannot work: every can would land in one
mask. This estimator segments by geometry instead, which the scene allows: the table is flat
and the cans are the only things standing on it.

1. Back-project every camera's depth into one world-frame cloud, inside a workspace box.
2. Fit the table plane to the low points (robustly) and measure every point's height above it.
   Heights come from this plane, not from world z, so a few mm of calibration error in z
   (measured: the table at z -9..0 mm through the 2026-11-27 calibration) does not reach the
   cans' heights. Reported z puts the table at 0, as the tasks do.
3. Per stack layer k (heights k*H .. (k+1)*H, H the can height), take the side-wall band of
   that layer and vote for can centres: each point votes for the centres one radius away
   (a circle Hough transform with the known radius). Peaks, suppressed within 1.5 radii,
   are the cans. Votes work with partial arcs and split cans that touch, which grouping by
   gaps cannot (cans in a 3-2-1 pyramid touch).
4. Each detection is refined with T6's `fit_cylinder` on its own points (top disc included).
5. Names: the cans are interchangeable, so the first frame assigns names in a fixed order
   (layer, then x, then y); later frames give each name the nearest detection to its
   previous estimate, so identities stay put while the skill moves cans one at a time.

Upright cans only (a can on its side is not detected). Tested on rendered sim frames of the
cylinder pyramid task and on one real frame of the six cans (2026-11-27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skillforge.contracts import Observation, TaskSpec
from skillforge.perception import CameraFrame, Estimate, backproject
from skillforge.perception.estimators import fit_cylinder, voxel_downsample, yaw_quat


@dataclass
class Detection:
    xy: np.ndarray  # world frame
    z: float  # centre height, table at 0
    layer: int
    votes: float
    n_points: int
    confidence: float


@dataclass
class IdenticalCylinderEstimator:
    """`PoseEstimator` for the task's movable cylinders, which must all have the same size."""

    workspace: tuple[float, float, float, float] = (0.15, 0.85, -0.45, 0.45)  # x0, x1, y0, y1 (m)
    min_height: float = 0.012  # m above the table plane to count as object
    voxel: float = 0.003
    vote_cell: float = 0.002  # m, accumulator resolution
    ring_tol: float = 0.004  # m, a point "on" a circle
    min_votes: float = 40.0
    rel_votes: float = 0.25  # of the strongest peak in the same layer
    min_fit: float = 0.3  # cylinder-fit confidence a weak peak needs to count as a can
    max_match: float = 0.08  # m: farther than this from its previous estimate, a name is not re-assigned
    needs_frames: bool = True
    names: list[str] = field(default_factory=list)
    radius: float = 0.0
    half_height: float = 0.0
    _last: dict[str, np.ndarray] = field(default_factory=dict)
    last_detections: list[Detection] = field(default_factory=list)
    last_table: tuple[float, float, float] | None = None  # plane z = c0 + c1 x + c2 y

    def reset(self, task: TaskSpec) -> None:
        cyl = [o for o in task.scene.objects if not o.fixed and o.shape == "cylinder"]
        if not cyl:
            raise ValueError("the task has no movable cylinders")
        sizes = {(round(o.size[0], 5), round(o.size[1], 5)) for o in cyl}
        if len(sizes) != 1:
            raise ValueError(f"identical cylinders expected, got sizes {sizes}")
        self.names = sorted(o.name for o in cyl)
        self.radius, self.half_height = float(cyl[0].size[0]), float(cyl[0].size[1])
        self._last = {}

    # -- the pipeline --------------------------------------------------------

    def _cloud(self, frames: list[CameraFrame]) -> np.ndarray:
        parts = [backproject(f) for f in frames if f is not None]
        if not parts:
            return np.zeros((0, 3))
        P = np.concatenate(parts)
        x0, x1, y0, y1 = self.workspace
        keep = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > -0.1) & (P[:, 2] < 0.6)
        return voxel_downsample(P[keep], self.voxel)

    @staticmethod
    def table_plane(P: np.ndarray) -> tuple[float, float, float]:
        """z = c0 + c1 x + c2 y through the table: least squares on low points, trimmed 6 times."""
        low = P[P[:, 2] < np.percentile(P[:, 2], 60)] if len(P) else P
        if len(low) < 50:
            return (0.0, 0.0, 0.0)
        A = np.c_[np.ones(len(low)), low[:, :2]]
        keep = np.ones(len(low), bool)
        c = np.zeros(3)
        for _ in range(6):
            c, *_ = np.linalg.lstsq(A[keep], low[keep, 2], rcond=None)
            r = low[:, 2] - A @ c
            keep = np.abs(r) < max(3 * np.std(r[keep]), 0.002)
        return float(c[0]), float(c[1]), float(c[2])

    def _votes(self, xy: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Circle-Hough accumulator: each point votes for centres at the known radius from it."""
        acc = np.zeros(shape)
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = self.radius * np.c_[np.cos(angles), np.sin(angles)]
        for dr in (-self.ring_tol / 2, 0.0, self.ring_tol / 2):  # a thick ring tolerates noise
            c = (xy[:, None, :] + ring[None] * (1 + dr / self.radius)).reshape(-1, 2)
            ij = np.floor((c - origin) / self.vote_cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
            np.add.at(acc, (ij[ok, 0], ij[ok, 1]), 1.0)
        # smooth with a 5x5 box so a peak is a neighbourhood, not one lucky cell
        k = 2
        pad = np.pad(acc, k)
        cs = pad.cumsum(0).cumsum(1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        n = 2 * k + 1
        return (cs[n:, n:] - cs[:-n, n:] - cs[n:, :-n] + cs[:-n, :-n]) / (n * n)

    def detect(self, frames: list[CameraFrame]) -> list[Detection]:
        P = self._cloud(frames)
        if len(P) == 0:
            self.last_table, self.last_detections = None, []
            return []
        c = self.table_plane(P)
        self.last_table = c
        h = P[:, 2] - (c[0] + c[1] * P[:, 0] + c[2] * P[:, 1])
        obj, hobj = P[h > self.min_height], h[h > self.min_height]
        H = 2 * self.half_height
        out: list[Detection] = []
        for layer in range(int(np.ceil(hobj.max() / H)) if len(hobj) else 0):
            band = (hobj > layer * H + 0.25 * H) & (hobj < layer * H + 0.8 * H)  # side wall of this layer only
            if band.sum() < self.min_votes:
                continue
            xy = obj[band, :2]
            origin = xy.min(axis=0) - 2 * self.radius
            shape = tuple(np.ceil((xy.max(axis=0) + 2 * self.radius - origin) / self.vote_cell).astype(int) + 1)
            acc = self._votes(xy, origin, shape)
            peaks: list[tuple[np.ndarray, float]] = []
            # Two upright cans in one layer cannot be closer than a diameter, so suppress that far.
            # (At 1.5 radii, the vote ring around a strong can left ghost peaks ~1.8 radii out.)
            sup = int(np.ceil(1.95 * self.radius / self.vote_cell))
            work = acc.copy()
            top = work.max()
            while True:
                i, j = np.unravel_index(int(np.argmax(work)), work.shape)
                v = work[i, j]
                if v < max(self.min_votes / 25.0, self.rel_votes * top):  # box-smoothed: /25 per cell
                    break
                peaks.append((origin + (np.array([i, j]) + 0.5) * self.vote_cell, float(v)))
                work[max(i - sup, 0):i + sup + 1, max(j - sup, 0):j + sup + 1] = 0.0
            for centre, v in peaks:
                near = np.linalg.norm(obj[:, :2] - centre, axis=1) < 1.3 * self.radius
                in_layer = (hobj > layer * H - 0.01) & (hobj < (layer + 1) * H + 0.015)
                sel = near & in_layer
                if sel.sum() < self.min_votes or hobj[sel].max() < layer * H + 0.6 * H:
                    continue  # not a can standing in this layer (a flat thing, or noise)
                pts = np.c_[obj[sel, :2], hobj[sel] - layer * H]  # this can as if it stood on the table
                fit = fit_cylinder(pts, self.radius, self.half_height)
                if fit.confidence < self.min_fit and v < 0.5 * top:
                    continue  # weak votes AND no cylinder shape: a ghost of a neighbour's vote ring
                xy_c = fit.xy if fit.xy is not None and fit.xy_confidence >= 0.3 else centre
                if np.linalg.norm(xy_c - centre) > self.radius:  # a fit that wandered off its peak
                    xy_c = centre
                out.append(Detection(np.asarray(xy_c, float), float(fit.pos[2] + layer * H), layer, v,
                                     int(sel.sum()), float(fit.confidence)))
        self.last_detections = out
        return out

    # -- PoseEstimator ---------------------------------------------------------

    def estimate(self, frames: list[CameraFrame], obs: Observation | None = None) -> dict[str, Estimate]:
        dets = self.detect(frames)
        pos = [np.array([d.xy[0], d.xy[1], d.z]) for d in dets]
        assigned: dict[str, int] = {}
        if not self._last:
            order = sorted(range(len(dets)), key=lambda k: (dets[k].layer, round(dets[k].xy[0], 2), dets[k].xy[1]))
            assigned = {n: k for n, k in zip(self.names, order, strict=False)}
        else:
            pairs = sorted((float(np.linalg.norm(pos[k] - self._last[n])), n, k)
                           for n in self.names if n in self._last for k in range(len(dets)))
            used: set[int] = set()
            for d, n, k in pairs:
                if n in assigned or k in used or d > self.max_match:
                    continue
                assigned[n], _ = k, used.add(k)
            free = [k for k in range(len(dets)) if k not in used]
            for n in self.names:  # names never seen, or lost: take what is left, in order
                if n not in assigned and free and n not in self._last:
                    assigned[n] = free.pop(0)
        out = {}
        for n in self.names:
            k = assigned.get(n)
            if k is None:
                out[n] = Estimate(np.zeros(3), yaw_quat(0.0), found=False)
                continue
            d = dets[k]
            self._last[n] = pos[k]
            out[n] = Estimate(pos[k].copy(), yaw_quat(0.0), True, confidence=d.confidence, n_points=d.n_points,
                              xy=d.xy.copy(), xy_confidence=d.confidence)
        return out
