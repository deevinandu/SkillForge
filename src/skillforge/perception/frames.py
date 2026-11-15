"""Camera frames: what a perception pipeline sees, the same in sim and on a real camera.

A `CameraFrame` is one RGB-D capture with its intrinsics and its pose in the world. Depth is
metric distance along the optical axis. The camera frame is the OpenCV one: +x right, +y down,
+z forward (into the scene). `T_world_cam` maps camera-frame points to the world frame.

`SimFrameSource` renders frames from a `MujocoEnv`. It reads the env's `model`/`data` and never
writes them. T7 implements `FrameSource` for real cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class CameraFrame:
    rgb: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW float32, metres along the optical axis; 0 = no return
    K: np.ndarray  # 3x3 pinhole intrinsics, pixels
    T_world_cam: np.ndarray  # 4x4, camera (OpenCV axes) -> world
    t: float = 0.0  # seconds since reset
    name: str = ""
    # Sim only: per-object boolean masks from the renderer's segmentation. Real frames leave
    # this empty and an estimator has to segment for itself.
    masks: dict[str, np.ndarray] = field(default_factory=dict)


@runtime_checkable
class FrameSource(Protocol):
    def capture(self, camera: str) -> CameraFrame | None:
        """One frame from `camera`, or None if it is unavailable."""


@dataclass(frozen=True)
class VirtualCamera:
    """A camera that is not in the env's model: MuJoCo's free camera, placed by orbit parameters.

    It lets perception place a camera without editing the scene. `azimuth` 180 puts the camera
    on the +x side of `lookat` looking back toward the robot; `elevation` is negative looking down.
    """

    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float  # degrees
    elevation: float  # degrees


def overview_camera(base_xy: tuple[float, float] = (0.0, 0.0)) -> VirtualCamera:
    """Default external camera: in front of the robot, 0.9 m from the workspace, looking down 50 deg."""
    return VirtualCamera("overview", (base_xy[0] + 0.5, base_xy[1], 0.02), 0.9, 180.0, -50.0)


def backproject(frame: CameraFrame, mask: np.ndarray | None = None) -> np.ndarray:
    """World-frame points (N, 3) for the pixels in `mask` (all pixels if None) with a depth return."""
    valid = frame.depth > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = frame.depth[v, u].astype(float)
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    pc = np.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z], axis=1)
    return pc @ frame.T_world_cam[:3, :3].T + frame.T_world_cam[:3, 3]


def project(frame: CameraFrame, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) -> pixel coordinates (N, 2) as (u, v), and depth (N,) along the axis."""
    R, t = frame.T_world_cam[:3, :3], frame.T_world_cam[:3, 3]
    pc = (np.asarray(points, dtype=float) - t) @ R
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = frame.K[0, 0] * pc[:, 0] / z + frame.K[0, 2]
        v = frame.K[1, 1] * pc[:, 1] / z + frame.K[1, 2]
    return np.stack([u, v], axis=1), z


class SimFrameSource:
    """Render RGB, depth and object masks from a `MujocoEnv`, for model or virtual cameras.

    `depth_noise` is the depth standard deviation in metres at 1 m; it grows with z squared, as a
    stereo depth camera's does. The noise stream is reseeded by `reseed(seed)`, so the same env
    seed gives the same frames.
    """

    def __init__(
        self,
        env,
        *,
        size: tuple[int, int] = (480, 640),  # (height, width)
        virtual: tuple[VirtualCamera, ...] = (),
        depth_noise: float = 0.0,
        max_depth: float = 4.0,
    ):
        import mujoco

        self._mj = mujoco
        self.env = env
        self.size = size
        self.virtual = {c.name: c for c in virtual}
        self.depth_noise = depth_noise
        self.max_depth = max_depth
        self._renderer = None
        self._failed = False
        self._rng = np.random.default_rng(0)
        m = env.model
        self._geom_to_obj = {int(m.geom(o.name).id): o.name for o in env.task.scene.objects}
        # Regions and the tcp marker are sites: drawing aids with no real-world counterpart. A
        # region box drawn around an object would also hide it from the segmentation image.
        self._opt = mujoco.MjvOption()
        self._opt.sitegroup[:] = 0

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng([seed, 2])

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _get_renderer(self):
        if self._renderer is None and not self._failed:
            # Let the env decide whether GL works: it knows the backend and never retries a
            # failed context (MuJoCo can abort the process on a second attempt).
            if self.env.render() is None:
                self._failed = True
                return None
            h, w = self.size
            self._renderer = self._mj.Renderer(self.env.model, h, w)
        return self._renderer

    def _camera(self, name: str):
        mj = self._mj
        if name in self.virtual:
            vc = self.virtual[name]
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = vc.lookat
            cam.distance, cam.azimuth, cam.elevation = vc.distance, vc.azimuth, vc.elevation
            return cam, float(self.env.model.vis.global_.fovy)
        cid = self.env.model.camera(name).id  # KeyError for an unknown name
        return name, float(self.env.model.cam_fovy[cid])

    def capture(self, camera: str) -> CameraFrame | None:
        r = self._get_renderer()
        if r is None:
            return None
        mj = self._mj
        cam, fovy = self._camera(camera)
        data = self.env.data
        h, w = self.size

        r.update_scene(data, camera=cam, scene_option=self._opt)
        rgb = r.render().copy()
        # The scene holds two GL cameras (left/right eye); the image is rendered from their mean.
        gl = r.scene.camera
        pos = (np.asarray(gl[0].pos) + np.asarray(gl[1].pos)) / 2
        fwd = (np.asarray(gl[0].forward) + np.asarray(gl[1].forward)) / 2
        up = (np.asarray(gl[0].up) + np.asarray(gl[1].up)) / 2
        fwd = fwd / np.linalg.norm(fwd)
        up = up - fwd * (up @ fwd)
        up = up / np.linalg.norm(up)
        right = np.cross(fwd, up)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, -up, fwd, pos
        f = (h / 2) / np.tan(np.radians(fovy) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])

        r.enable_depth_rendering()
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        depth[depth > self.max_depth] = 0.0
        if self.depth_noise > 0:
            noise = self._rng.normal(0.0, 1.0, depth.shape).astype(np.float32)
            depth = np.where(depth > 0, depth + self.depth_noise * depth * depth * noise, 0.0).astype(np.float32)

        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        is_geom = seg[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM)
        masks = {name: is_geom & (seg[:, :, 0] == gid) for gid, name in self._geom_to_obj.items()}

        return CameraFrame(rgb=rgb, depth=depth, K=K, T_world_cam=T, t=float(data.time), name=camera, masks=masks)
