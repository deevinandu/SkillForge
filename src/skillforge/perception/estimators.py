"""Object pose estimators: camera frames in, one `Estimate` per object out.

`OracleEstimator` passes through what the env already reports (true pose plus the env's own
observation noise): today's behaviour, and the reference for every comparison.
`DepthEstimator` is classical and numpy-only: segment the object, back-project its depth pixels
from every camera into one world-frame cloud, and fit the known `ObjectSpec` geometry to it.

Assumptions of the geometric fit (stated, not hidden): a box rests upright on a horizontal
surface or hangs upright in the gripper, so its top surface is horizontal; a cylinder stands
upright or lies on its side. A box tipped onto an edge, a cylinder at some other tilt, or an
object half hidden by the fingers does not match: it gets a coarse estimate (the centre of the
visible points) and a low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from skillforge.contracts import ObjectSpec, Observation, TaskSpec

from .frames import CameraFrame, backproject


@dataclass
class Estimate:
    pos: np.ndarray  # (3,) world frame
    quat: np.ndarray  # (4,) w, x, y, z
    found: bool  # False: too few pixels on the object; pos/quat mean nothing
    # 0..1, how well the cloud agrees with the known upright geometry. Below the estimator's
    # `min_confidence` the pose is coarse: the centre of the visible points, not a model fit.
    confidence: float = 0.0
    n_points: int = 0
    # Horizontal position from a part of the fit that survives a hidden top (a cylinder's side
    # wall), with its own confidence. None for shapes without such a fit.
    xy: np.ndarray | None = None
    xy_confidence: float = 0.0


@runtime_checkable
class PoseEstimator(Protocol):
    needs_frames: bool  # False lets the env skip rendering

    def reset(self, task: TaskSpec) -> None: ...

    def estimate(self, frames: list[CameraFrame], obs: Observation) -> dict[str, Estimate]:
        """One Estimate per movable object the estimator knows about."""


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(angle: float, period: float) -> float:
    """Wrap into (-period/2, period/2]."""
    a = (angle + period / 2) % period - period / 2
    return a + period if a <= -period / 2 else a


# ---------------------------------------------------------------- oracle


class OracleEstimator:
    """What the env reports, optionally with extra Gaussian position noise (metres)."""

    needs_frames = False

    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng([seed, 3])

    def reset(self, task: TaskSpec) -> None:
        pass

    def estimate(self, frames: list[CameraFrame], obs: Observation) -> dict[str, Estimate]:
        out = {}
        for name, o in obs.objects.items():
            p = np.asarray(o.pos, dtype=float).copy()
            if self.noise_std > 0:
                p = p + self._rng.normal(0.0, self.noise_std, 3)
            out[name] = Estimate(pos=p, quat=np.asarray(o.quat, dtype=float).copy(), found=True, confidence=1.0)
        return out


# ---------------------------------------------------------------- geometric fits


def _top_points(pts: np.ndarray, band: float) -> tuple[np.ndarray, float]:
    """Points on the top surface and its height: a robust maximum of z, then a band below it."""
    z_top = float(np.percentile(pts[:, 2], 97))
    top = pts[pts[:, 2] > z_top - band]
    return top, float(np.median(top[:, 2]))


def _min_area_rect(xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Smallest enclosing rectangle of 2D points: (centre, angle of its first axis, full extents).

    Brute force over the quarter turn (a rectangle repeats every 90 degrees), then a fine pass.
    Extents use the 1st and 99th percentiles, so a few stray pixels do not stretch the box.
    """

    def measure(th: float):
        c, s = np.cos(th), np.sin(th)
        a = xy @ np.array([c, s])
        b = xy @ np.array([-s, c])
        a0, a1 = np.percentile(a, [1, 99])
        b0, b1 = np.percentile(b, [1, 99])
        return (a1 - a0) * (b1 - b0), (a0 + a1) / 2, (b0 + b1) / 2, a1 - a0, b1 - b0

    coarse = np.radians(np.arange(0.0, 90.0, 2.0))
    best = min(coarse, key=lambda th: measure(th)[0])
    fine = best + np.radians(np.arange(-2.0, 2.01, 0.25))
    th = float(min(fine, key=lambda t: measure(t)[0]))
    _, ca, cb, ea, eb = measure(th)
    c, s = np.cos(th), np.sin(th)
    centre = np.array([c * ca - s * cb, s * ca + c * cb])
    return centre, th, np.array([ea, eb])


def _extent_score(measured: np.ndarray, expected: np.ndarray) -> float:
    """1 when the measured extents equal the expected ones, falling to 0 at 50% relative error."""
    rel = np.max(np.abs(np.sort(measured) - np.sort(expected)) / np.maximum(np.sort(expected), 1e-6))
    return float(np.clip(1.0 - 2.0 * rel, 0.0, 1.0))


def _span_score(pts: np.ndarray, height: float) -> float:
    """Does the cloud span the object's full height? A camera looking down at an angle sees the
    side walls of an upright object all the way down; a cylinder on its side, or an object whose
    lower half is hidden, spans less."""
    lo, hi = np.percentile(pts[:, 2], [1, 99])
    return float(np.clip(1.0 - 2.0 * abs((hi - lo) - height) / height, 0.0, 1.0))


def fit_box(pts: np.ndarray, half: np.ndarray, prev_yaw: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Upright box with half-extents `half`: (centre, quat, extent agreement)."""
    top, z_top = _top_points(pts, band=max(0.004, 0.2 * half[2]))
    centre_xy, th, ext = _min_area_rect(top[:, :2])
    expected = 2 * half[:2]
    square = abs(half[0] - half[1]) < 0.1 * max(half[0], half[1])
    if square:
        period = np.pi / 2
    else:
        period = np.pi
        # `th` is the rectangle's first axis; turn it so it is the box's local x
        if (ext[0] > ext[1]) != (half[0] > half[1]):
            th += np.pi / 2
    # a symmetric box has several equivalent yaws: report the one nearest the last estimate
    ref = 0.0 if prev_yaw is None else prev_yaw
    yaw = ref + wrap(th - ref, period)
    centre = np.array([centre_xy[0], centre_xy[1], z_top - half[2]])
    return centre, yaw_quat(yaw), min(_extent_score(ext, expected), _span_score(pts, 2 * half[2]))


def _fit_centre(pts: np.ndarray, radius: float, init: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre of a circle (2D points) or sphere (3D) of known radius: Gauss-Newton on
    (|p - c| - r)^2 from `init`. Returns (centre, RMS of the residual). A partial arc is enough."""
    c = np.asarray(init, dtype=float).copy()
    for _ in range(10):
        d = pts - c
        n = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        step, *_ = np.linalg.lstsq(-d / n[:, None], -(n - radius), rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - radius) ** 2)))
    return c, rms


def _round_score(rms: float, radius: float) -> float:
    """1 for points exactly on the circle/sphere, 0 once the RMS residual reaches a quarter radius."""
    return float(np.clip(1.0 - rms / (0.25 * radius), 0.0, 1.0))


@dataclass
class CylinderFit:
    pos: np.ndarray
    quat: np.ndarray
    confidence: float
    lying: bool  # True: the axis is horizontal (the cylinder is on its side)
    # Horizontal centre from the side wall alone, assuming a vertical axis. It does not need the
    # top to be visible, so it still works when the fingers cover the top of a held cylinder.
    xy: np.ndarray | None
    xy_confidence: float


def fit_cylinder(pts: np.ndarray, radius: float, half_height: float) -> CylinderFit:
    """Cylinder of known size, standing upright or lying on its side: whichever fits better.

    Upright: the side wall is a circle of the known radius seen from above (fitted even from a
    partial arc); the top disc's height gives z. Lying: the cross-section perpendicular to the
    long horizontal axis is that circle, which gives the centre across the axis and in z.
    """
    band = max(0.004, 0.2 * half_height)
    top, z_top = _top_points(pts, band)
    lo, hi = np.percentile(top[:, :2], [1, 99], axis=0)
    top_centre = (lo + hi) / 2
    wall = pts[pts[:, 2] < z_top - band]
    xy, xy_conf = None, 0.0
    if len(wall) >= 30:
        xy, rms = _fit_centre(wall[:, :2], radius, top_centre)
        xy_conf = _round_score(rms, radius)
    up_conf = min(_extent_score(hi - lo, np.array([2 * radius, 2 * radius])), _span_score(pts, 2 * half_height))
    up_xy = xy if xy is not None and xy_conf >= up_conf else top_centre
    upright = CylinderFit(
        np.array([up_xy[0], up_xy[1], z_top - half_height]), yaw_quat(0.0), up_conf, False, xy, xy_conf
    )

    centre_xy, th, ext = _min_area_rect(pts[:, :2])
    if ext[1] > ext[0]:
        th, ext = th + np.pi / 2, ext[::-1]
    axis, perp = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    t = pts[:, :2] @ axis
    along = np.percentile(t, [1, 99])
    # the end caps are flat discs, not part of the round wall: fit the circle to the middle
    margin = 0.1 * (along[1] - along[0])
    mid = pts[(t > along[0] + margin) & (t < along[1] - margin)]
    if len(mid) < 30:
        return upright
    section = np.column_stack([mid[:, :2] @ perp, mid[:, 2]])
    c, rms = _fit_centre(section, radius, np.array([centre_xy @ perp, np.percentile(mid[:, 2], 97) - radius]))
    length_score = float(np.clip(1.0 - 2.0 * abs((along[1] - along[0]) - 2 * half_height) / (2 * half_height), 0.0, 1.0))
    # No height-span test here: a camera looking down at an angle never sees the underside.
    lying_conf = min(_round_score(rms, radius), length_score)
    if lying_conf <= max(up_conf, 0.5):  # a weak "lying" guess must not replace the upright fit
        return upright
    pos_xy = axis * along.mean() + perp * c[0]
    # object z (the cylinder's axis) laid along `axis`: tip it onto world x, then yaw to th
    tip = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    w1, x1, y1, z1 = yaw_quat(th)
    w2, x2, y2, z2 = tip
    quat = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return CylinderFit(np.array([pos_xy[0], pos_xy[1], c[1]]), quat, lying_conf, True, None, 0.0)


def fit_sphere(pts: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Sphere of known radius, from whatever cap of it is visible."""
    init = pts.mean(axis=0)
    init[2] = np.percentile(pts[:, 2], 97) - radius
    c, rms = _fit_centre(pts, radius, init)
    return c, yaw_quat(0.0), _round_score(rms, radius)


# ---------------------------------------------------------------- depth estimator


def color_mask(rgb: np.ndarray, target: tuple[float, float, float], hue_tol: float = 0.06, min_sat: float = 0.35) -> np.ndarray:
    """Pixels whose hue is within `hue_tol` (fraction of the colour wheel) of `target` (rgb 0..1)."""

    def hue_sat(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mx, mn = c.max(axis=-1), c.min(axis=-1)
        d = mx - mn
        r, g, b = c[..., 0], c[..., 1], c[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
            s = np.where(mx > 0, d / mx, 0.0)
        return np.where(d > 0, h, 0.0), s

    h, s = hue_sat(rgb.astype(float) / 255.0)
    ht, _ = hue_sat(np.asarray(target, dtype=float))
    dh = np.abs(h - ht)
    return (np.minimum(dh, 1 - dh) < hue_tol) & (s > min_sat)


def voxel_downsample(pts: np.ndarray, size: float) -> np.ndarray:
    """One point (the mean) per occupied voxel of edge `size`. Deterministic."""
    if size <= 0 or len(pts) == 0:
        return pts
    k = np.floor((pts - pts.min(axis=0)) / size).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]  # fine below 2^21 voxels per axis
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    return out / counts[:, None]


class DepthEstimator:
    """Fit known object geometry to a fused depth cloud.

    segment='mask' uses the renderer's per-object masks (sim only: perfect segmentation, so the
    measured error is geometry, occlusion, resolution and depth noise alone). segment='color'
    segments by hue, which also works on a real camera; `colors` maps object name -> rgb (0..1).
    """

    needs_frames = True

    def __init__(
        self,
        segment: Literal["mask", "color"] = "mask",
        colors: dict[str, tuple[float, float, float]] | None = None,
        min_points: int = 40,
        # Measured on the cylinder task: fits scoring 0.8+ are within ~1 mm, fits at 0.3-0.6 are
        # ~5 mm off (the fingers already hide part of the object). Below this the pose is coarse.
        min_confidence: float = 0.7,
        table_z: float = 0.0,
        voxel: float = 0.002,  # metres; evens out density so a close camera does not outvote a far one
        max_pixels: int = 4000,  # per camera and object
    ):
        if segment == "color" and not colors:
            raise ValueError("segment='color' needs `colors`: object name -> rgb")
        self.segment = segment
        self.colors = colors or {}
        self.min_points = min_points
        self.min_confidence = min_confidence
        self.table_z = table_z
        self.voxel = voxel
        self.max_pixels = max_pixels
        self._objects: dict[str, ObjectSpec] = {}
        self._yaw: dict[str, float] = {}

    def reset(self, task: TaskSpec) -> None:
        self._objects = {o.name: o for o in task.scene.objects if not o.fixed}
        self._yaw = {}

    def _cloud(self, name: str, frames: list[CameraFrame]) -> np.ndarray:
        parts = []
        for f in frames:
            if self.segment == "mask":
                m = f.masks.get(name)
                if m is None:
                    continue
            else:
                if name not in self.colors:
                    continue
                m = color_mask(f.rgb, self.colors[name])
            n = int(m.sum())
            if n > self.max_pixels:  # a close camera: keep an even, deterministic subset of its pixels
                idx = np.flatnonzero(m)[:: int(np.ceil(n / self.max_pixels))]
                m = np.zeros(m.size, dtype=bool)
                m[idx] = True
                m = m.reshape(f.depth.shape)
            p = backproject(f, m)
            if self.segment == "color":
                p = p[p[:, 2] > self.table_z + 0.003]  # same-hue pixels on the table are not the object
            parts.append(p)
        if not parts:
            return np.zeros((0, 3))
        return voxel_downsample(np.concatenate(parts), self.voxel)

    def estimate(self, frames: list[CameraFrame], obs: Observation) -> dict[str, Estimate]:
        out = {}
        for name, o in self._objects.items():
            pts = self._cloud(name, frames)
            xy, xy_conf = None, 0.0
            if len(pts) < self.min_points:
                out[name] = Estimate(np.zeros(3), yaw_quat(0.0), found=False, n_points=len(pts))
                continue
            if o.shape == "box":
                pos, quat, conf = fit_box(pts, np.asarray(o.size, dtype=float), self._yaw.get(name))
            elif o.shape == "cylinder":
                fit = fit_cylinder(pts, o.size[0], o.size[1])
                pos, quat, conf, xy, xy_conf = fit.pos, fit.quat, fit.confidence, fit.xy, fit.xy_confidence
            elif o.shape == "sphere":
                pos, quat, conf = fit_sphere(pts, o.size[0])
            else:  # mesh: no model fit yet, so always the coarse estimate below
                pos, quat, conf = pts.mean(axis=0), yaw_quat(0.0), 0.0
            if conf < self.min_confidence:
                # The cloud does not look like the known shape (tipped over, or half hidden).
                # The middle of what is visible is still far better than a stale pose.
                lo, hi = np.percentile(pts, [1, 99], axis=0)
                pos, quat = (lo + hi) / 2, yaw_quat(self._yaw.get(name, 0.0))
            elif o.shape == "box":
                self._yaw[name] = quat_yaw(quat)
            out[name] = Estimate(pos, quat, True, confidence=conf, n_points=len(pts), xy=xy, xy_confidence=xy_conf)
        return out
