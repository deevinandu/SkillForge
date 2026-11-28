"""T7: a `FrameSource` over Intel RealSense depth cameras, through `pyrealsense2` (no ROS).

Each camera streams colour and depth. Depth is aligned to the colour image, so both share one
pixel grid and the colour intrinsics `K`, and is converted to metres along the optical axis
(0 = no return), which is what T6's `CameraFrame` asks for. Colour is RGB, not BGR, because
`color_mask` assumes RGB.

`T_world_cam` cannot come from the camera: it is a calibration, loaded with `load_extrinsics`.
The world frame is the robot base frame (tasks leave `robot.base_pose` at the origin), so a
base<-camera transform from a hand-eye calibration is used as is.

The camera model is pinhole. Colour streams that report non-zero distortion get a warning,
since `project`/`backproject` ignore it. A D415's colour stream reports zero distortion.

`pyrealsense2` is imported only when a source is created without an injected module, so the rest
of `skillforge.real` and the tests never need it.

Run against a D415 on the lab workstation (2026-11-27). Not yet used inside a `RealEnv` session.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skillforge.perception import CameraFrame


@dataclass(frozen=True, eq=False)
class RealSenseCamera:
    """One physical camera, addressed by `name` in `capture(name)`."""

    name: str
    T_world_cam: np.ndarray  # 4x4, camera (OpenCV axes) -> world (robot base)
    serial: str | None = None  # None: the only connected camera
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30  # dropped at start while auto-exposure settles
    timeout_ms: int = 1000
    max_depth: float = 4.0  # metres; farther returns become 0

    def __post_init__(self):
        object.__setattr__(self, "T_world_cam", check_transform(self.T_world_cam))


def check_transform(T) -> np.ndarray:
    """`T` as a float 4x4 rigid transform, or ValueError."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_cam must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"T_world_cam's last row must be [0, 0, 0, 1], got {T[3]}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0.99:
        raise ValueError("T_world_cam's rotation is not a rotation (R^T R != I or det(R) != 1)")
    return T


def load_extrinsics(path: str | Path) -> np.ndarray:
    """A world<-camera transform from `.npy` (a 4x4 array, as hand-eye scripts save it) or `.json`
    (`{"T_world_cam": [[...], ...]}` or a bare 4x4 list)."""
    p = Path(path)
    if p.suffix == ".npy":
        return check_transform(np.load(p))
    data = json.loads(p.read_text())
    return check_transform(data["T_world_cam"] if isinstance(data, dict) else data)


def save_extrinsics(path: str | Path, T: np.ndarray, **meta) -> None:
    """Write `T` as json, with free-form metadata (how it was calibrated, when, which serial)."""
    Path(path).write_text(json.dumps({"T_world_cam": check_transform(T).tolist(), **meta}, indent=2))


def _pyrealsense2():
    try:
        import pyrealsense2 as rs
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise ImportError("RealSense cameras need pyrealsense2: pip install pyrealsense2") from e
    return rs


def list_devices(rs=None) -> list[tuple[str, str]]:
    """(name, serial) of every connected RealSense device."""
    rs = rs or _pyrealsense2()
    return [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().devices]


@dataclass
class _Stream:
    cam: RealSenseCamera
    pipeline: object
    align: object
    depth_scale: float
    K: np.ndarray
    serial: str
    distortion: tuple[float, ...] = field(default_factory=tuple)


class RealSenseFrameSource:
    """`FrameSource` for one or more RealSense cameras. Streams start on `start()` or first capture.

    `capture(name)` waits up to the camera's `timeout_ms` for the next frame pair and returns
    None if none comes (unplugged, USB stall). An unknown name is a KeyError, as in sim.
    `t` on each frame is seconds since construction or the last `reset_clock()`.
    """

    def __init__(self, cameras: Iterable[RealSenseCamera], *, rs=None, clock: Callable[[], float] = time.monotonic):
        self.cameras = {c.name: c for c in cameras}
        if not self.cameras:
            raise ValueError("give at least one RealSenseCamera")
        self._rs = rs
        self._clock = clock
        self._t0 = clock()
        self._streams: dict[str, _Stream] = {}

    @property
    def rs(self):
        if self._rs is None:
            self._rs = _pyrealsense2()
        return self._rs

    def reset_clock(self) -> None:
        self._t0 = self._clock()

    def start(self) -> None:
        for name in self.cameras:
            self._stream(name)

    def _stream(self, name: str) -> _Stream:
        s = self._streams.get(name)
        if s is not None:
            return s
        cam, rs = self.cameras[name], self.rs
        config = rs.config()
        if cam.serial:
            config.enable_device(cam.serial)
        config.enable_stream(rs.stream.color, cam.width, cam.height, rs.format.rgb8, cam.fps)
        config.enable_stream(rs.stream.depth, cam.width, cam.height, rs.format.z16, cam.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        try:
            device = profile.get_device()
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
            distortion = tuple(float(c) for c in intr.coeffs)
            if any(abs(c) > 1e-9 for c in distortion):
                warnings.warn(f"camera {name!r}: colour stream reports distortion {distortion}; "
                              "CameraFrame is pinhole, so points near the image edge will be off", stacklevel=2)
            serial = str(device.get_info(rs.camera_info.serial_number))
            for _ in range(cam.warmup_frames):
                pipeline.wait_for_frames(cam.timeout_ms)
        except Exception:
            pipeline.stop()
            raise
        s = _Stream(cam, pipeline, rs.align(rs.stream.color), depth_scale, K, serial, distortion)
        self._streams[name] = s
        return s

    def capture(self, camera: str) -> CameraFrame | None:
        if camera not in self.cameras:
            raise KeyError(camera)
        s = self._stream(camera)
        ok, frames = s.pipeline.try_wait_for_frames(s.cam.timeout_ms)
        if not ok:
            return None
        frames = s.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        rgb = np.array(np.asanyarray(color.get_data())[..., :3], dtype=np.uint8, copy=True)
        d = np.asanyarray(depth.get_data()).astype(np.float32) * np.float32(s.depth_scale)
        d[d > s.cam.max_depth] = 0.0
        return CameraFrame(rgb=rgb, depth=d, K=s.K.copy(), T_world_cam=s.cam.T_world_cam.copy(),
                           t=float(self._clock() - self._t0), name=camera)

    def info(self, camera: str) -> dict:
        """Serial, intrinsics and depth scale of a started camera, for logs."""
        s = self._stream(camera)
        return {"name": camera, "serial": s.serial, "size": [s.cam.height, s.cam.width], "K": s.K.tolist(),
                "depth_scale": s.depth_scale, "distortion": list(s.distortion)}

    def close(self) -> None:
        for s in self._streams.values():
            try:
                s.pipeline.stop()
            except Exception:
                pass
        self._streams.clear()

    def __enter__(self) -> RealSenseFrameSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ checking a calibration

AXIS_COLORS = ((240, 40, 40), (40, 220, 60), (60, 90, 250))  # x red, y green, z blue


def calibration_overlay(frame: CameraFrame, axis_len: float = 0.15, grid: tuple[float, float, float, float] = (0.3, 0.7, -0.3, 0.3),
                        grid_step: float = 0.1, table_z: float = 0.0, width: int = 2) -> np.ndarray:
    """A copy of `frame.rgb` with the robot-base axes and a table-plane grid drawn through `T_world_cam`.

    If the calibration is right, the axes start at the robot's base (x forward, y left, z up) and
    the grid lies flat on the table in front of it, with lines every `grid_step` metres over
    x in [grid[0], grid[1]] and y in [grid[2], grid[3]].
    """
    from .overlay import _polyline

    img = np.array(frame.rgb[..., :3], dtype=np.uint8, copy=True)
    x0, x1, y0, y1 = grid
    for x in np.arange(x0, x1 + 1e-9, grid_step):
        _polyline(img, frame, np.array([[x, y0, table_z], [x, y1, table_z]]), (230, 230, 230), 1)
    for y in np.arange(y0, y1 + 1e-9, grid_step):
        _polyline(img, frame, np.array([[x0, y, table_z], [x1, y, table_z]]), (230, 230, 230), 1)
    for axis, color in zip(np.eye(3), AXIS_COLORS, strict=True):
        _polyline(img, frame, np.array([[0.0, 0.0, 0.0], axis * axis_len]), color, width)
    return img
