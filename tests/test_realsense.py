"""T7: the RealSense FrameSource against a fake pyrealsense2. The last test needs a real camera and skips without one."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from skillforge.perception import CameraFrame, FrameSource, project
from skillforge.real import (
    RealSenseCamera,
    RealSenseFrameSource,
    calibration_overlay,
    list_devices,
    load_extrinsics,
    save_extrinsics,
)

H, W = 48, 64
K = (60.0, 61.0, 31.5, 23.5)  # fx, fy, cx, cy


def look_down(height: float = 1.0, x: float = 0.5) -> np.ndarray:
    """Camera above (x, 0, height) looking straight down: OpenCV z = -world z, x = -world y, y = -world x."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
    T[:3, 3] = [x, 0.0, height]
    return T


class FakeRS:
    """Just enough of pyrealsense2 for RealSenseFrameSource: one camera, depth in millimetres."""

    def __init__(self, serial="1234", depth_mm=800, distortion=(0.0,) * 5, drop_after=None):
        self.serial, self.depth_mm, self.distortion, self.drop_after = serial, depth_mm, distortion, drop_after
        self.stream = SimpleNamespace(color="color", depth="depth")
        self.format = SimpleNamespace(rgb8="rgb8", z16="z16")
        self.camera_info = SimpleNamespace(name="name", serial_number="serial")
        self.enabled_serials, self.streams, self.stopped, self.waits, self.n_frames = [], [], 0, 0, 0
        rs = self

        class Config:
            def enable_device(self, s):
                rs.enabled_serials.append(s)

            def enable_stream(self, *args):
                rs.streams.append(args)

        class Device:
            def get_info(self, key):
                return {"name": "Intel RealSense D415", "serial": rs.serial}[key]

            def first_depth_sensor(self):
                return SimpleNamespace(get_depth_scale=lambda: 0.001)

        intr = SimpleNamespace(fx=K[0], fy=K[1], ppx=K[2], ppy=K[3], coeffs=list(distortion))

        class Profile:
            def get_device(self):
                return Device()

            def get_stream(self, s):
                return SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))

        def frameset():
            rgb = np.zeros((H, W, 3), np.uint8)
            rgb[..., 0] = 200
            depth = np.full((H, W), rs.depth_mm, np.uint16)
            depth[0, 0] = 9000  # 9 m: beyond max_depth
            return SimpleNamespace(get_color_frame=lambda: SimpleNamespace(get_data=lambda: rgb),
                                   get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth))

        class Pipeline:
            def start(self, config):
                return Profile()

            def stop(self):
                rs.stopped += 1

            def wait_for_frames(self, timeout):
                rs.waits += 1
                return frameset()

            def try_wait_for_frames(self, timeout):
                rs.n_frames += 1
                if rs.drop_after is not None and rs.n_frames > rs.drop_after:
                    return False, None
                return True, frameset()

        self.config, self.pipeline = Config, Pipeline
        self.align = lambda to: SimpleNamespace(process=lambda f: f)
        self.context = lambda: SimpleNamespace(devices=[Device()])


def source(rs=None, **cam_kw):
    rs = rs or FakeRS()
    cam = RealSenseCamera("overview", look_down(), **{"warmup_frames": 3, **cam_kw})
    return RealSenseFrameSource([cam], rs=rs, clock=iter(np.arange(0.0, 100.0, 0.5)).__next__), rs


def test_capture_gives_a_camera_frame_in_metres():
    src, rs = source()
    assert isinstance(src, FrameSource)
    f = src.capture("overview")
    assert isinstance(f, CameraFrame) and f.name == "overview"
    assert f.rgb.shape == (H, W, 3) and f.rgb.dtype == np.uint8 and f.rgb[0, 0, 0] == 200
    assert f.depth.dtype == np.float32 and f.depth[5, 5] == pytest.approx(0.8)
    assert f.depth[0, 0] == 0.0  # past max_depth: no return
    np.testing.assert_allclose(f.K, [[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
    np.testing.assert_allclose(f.T_world_cam, look_down())
    assert f.masks == {}  # real frames: the estimator segments for itself
    assert rs.waits == 3  # warm-up frames dropped
    assert rs.streams == [("color", 640, 480, "rgb8", 30), ("depth", 640, 480, "z16", 30)]  # RGB, not BGR
    src.close()
    assert rs.stopped == 1


def test_frame_time_counts_from_the_last_clock_reset():
    src, _ = source()  # clock ticks 0.5 s per call; construction read 0.0
    t1 = src.capture("overview").t
    src.reset_clock()
    t2 = src.capture("overview").t
    assert t1 > 0 and t2 == pytest.approx(0.5)


def test_a_missing_frame_is_none_and_an_unknown_camera_is_a_key_error():
    src, _ = source(FakeRS(drop_after=1))
    assert src.capture("overview") is not None
    assert src.capture("overview") is None
    with pytest.raises(KeyError):
        src.capture("wrist")


def test_serial_selects_the_device_and_info_reports_it():
    src, rs = source(serial="1234")
    info = src.info("overview")
    assert rs.enabled_serials == ["1234"]
    assert info["serial"] == "1234" and info["depth_scale"] == 0.001 and info["size"] == [480, 640]
    assert list_devices(rs) == [("Intel RealSense D415", "1234")]


def test_distortion_is_warned_about():
    src, _ = source(FakeRS(distortion=(0.1, 0, 0, 0, 0)))
    with pytest.warns(UserWarning, match="distortion"):
        src.start()


def test_depth_backprojects_onto_the_table_through_the_extrinsics():
    from skillforge.perception import backproject

    src, _ = source(FakeRS(depth_mm=1000))  # camera 1 m above the table, so every point lands at z = 0
    f = src.capture("overview")
    f.depth[0, 0] = 1.0
    pts = backproject(f)
    np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-6)
    centre = backproject(f, np.pad(np.ones((1, 1), bool), ((23, H - 24), (31, W - 32))))
    np.testing.assert_allclose(centre[0, :2], [0.5, 0.0], atol=0.02)


@pytest.mark.parametrize("T", [np.eye(3), np.diag([1.0, 1.0, -1.0, 1.0]), np.vstack([np.eye(4)[:3], [1, 0, 0, 1]])])
def test_bad_extrinsics_are_rejected(T):
    with pytest.raises(ValueError):
        RealSenseCamera("x", T)


def test_extrinsics_round_trip_through_npy_and_json(tmp_path):
    T = look_down(0.7, 0.4)
    np.save(tmp_path / "T.npy", T)
    np.testing.assert_allclose(load_extrinsics(tmp_path / "T.npy"), T)
    save_extrinsics(tmp_path / "T.json", T, method="aruco hand-eye", serial="1234")
    np.testing.assert_allclose(load_extrinsics(tmp_path / "T.json"), T)
    assert json.loads((tmp_path / "T.json").read_text())["method"] == "aruco hand-eye"
    (tmp_path / "bare.json").write_text(json.dumps(T.tolist()))
    np.testing.assert_allclose(load_extrinsics(tmp_path / "bare.json"), T)


def test_calibration_overlay_draws_the_base_axes_where_they_project():
    T = look_down(1.0, x=0.3)  # base origin is 0.3 m off the optical axis, inside the image
    Kf = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    f = CameraFrame(rgb=np.zeros((H, W, 3), np.uint8), depth=np.ones((H, W), np.float32), K=Kf, T_world_cam=T)
    img = calibration_overlay(f, axis_len=0.1, grid=(0.25, 0.35, -0.05, 0.05), grid_step=0.05)
    ux, vx = project(f, np.array([[0.05, 0, 0]]))[0][0]  # halfway along +x: red
    assert tuple(img[int(round(vx)), int(round(ux))]) == (240, 40, 40)
    assert img.sum() > 0 and f.rgb.sum() == 0  # drew on a copy


def test_live_camera_if_connected():
    """Hardware: runs only where pyrealsense2 is installed and a camera is plugged in."""
    rs = pytest.importorskip("pyrealsense2")
    if not list_devices(rs):
        pytest.skip("no RealSense camera connected")
    with RealSenseFrameSource([RealSenseCamera("overview", np.eye(4))]) as src:
        f = src.capture("overview")
    assert f is not None and f.rgb.shape == (480, 640, 3) and f.depth.shape == (480, 640)
    valid = f.depth[f.depth > 0]
    assert valid.size > 0.2 * f.depth.size and 0.1 < np.median(valid) < 4.0
