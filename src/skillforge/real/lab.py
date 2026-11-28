"""T7: the lab cell (Franka + frankapy bridge + fixed RealSense) assembled into a `RealEnv`.

    env = make_lab_env(task, extrinsics="runs/real/handeye/2026-11-27/T_world_cam.json")

- arm: `FrankaBridgeRobotIO` (the bridge server must be running, see scripts/frankapy_bridge_server.py);
- camera: one fixed `RealSenseFrameSource` camera named "fixed", posed by the hand-eye calibration;
- objects: `IdenticalCylinderEstimator` (the lab's cans), the only estimator wired here so far;
- controller: the sim's `MujocoController`, so the IK is the one the skill was evaluated with;
- operator: `TerminalOperator`, which also saves `placement.png` (targets green, estimates red)
  before each prompt;
- `realtime=True`: steps are paced at the task's control rate.

Each part is replaceable by keyword for tests. Nothing here moves the arm by itself: the arm
moves when the caller resets or steps the env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skillforge.contracts import TaskSpec
from skillforge.perception import CameraFrame

from .cans import IdenticalCylinderEstimator
from .controller import MujocoController
from .core import Placement, RealEnv
from .franka_bridge import FrankaBridgeRobotIO
from .overlay import placement_overlay
from .realsense import RealSenseCamera, RealSenseFrameSource, load_extrinsics
from .runner import TerminalOperator
from .safety import SafetyLimits

CAMERA = "fixed"


def identical_cylinders(task: TaskSpec) -> bool:
    movable = [o for o in task.scene.objects if not o.fixed]
    return bool(movable) and all(o.shape == "cylinder" for o in movable) and \
        len({(round(o.size[0], 5), round(o.size[1], 5)) for o in movable}) == 1


class PlacementPicture:
    """`TerminalOperator.show`: save the camera image with every target (green) and estimate (red)."""

    def __init__(self, task: TaskSpec, frames, estimator, path: str | Path):
        self.task, self.frames, self.estimator, self.path = task, frames, estimator, Path(path)

    def __call__(self, seed: int, targets: dict[str, Placement]) -> None:
        frame = self.frames.capture(CAMERA)
        if frame is None:
            return
        est = self.estimator.estimate([frame])
        img = frame.rgb
        for name, t in targets.items():
            img = placement_overlay(CameraFrame(rgb=img, depth=frame.depth, K=frame.K, T_world_cam=frame.T_world_cam),
                                    self.task.scene.object(name), t, est.get(name))
        try:
            import imageio.v3 as iio
        except ImportError:  # pragma: no cover
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(self.path, np.asarray(img))


class Preview:
    """While a person arranges objects: refresh `path` about `hz` times a second with the camera
    image, the start area (green), the goal spots on the table (orange), and a circle on each
    detected object, green if fine and red if `check_layout` names it, with the problems written
    at the top. Uses its OWN estimator, fresh each frame, so it never changes the session's
    naming. `stop()` joins the thread before anything else uses the camera."""

    def __init__(self, task: TaskSpec, frames, path: str | Path, hz: float = 2.0):
        from .free_placement import LAB_START_AREA

        self.task, self.frames, self.path, self.period = task, frames, Path(path), 1.0 / hz
        self.area = LAB_START_AREA
        self._thread = None
        self._stop = None
        self.frames_written = 0

    def start(self) -> None:
        import threading

        self.stop()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        import time

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.render_once()
            except Exception as e:  # a preview must never take the session down
                print(f"(preview: {type(e).__name__}: {e})")
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def render_once(self) -> None:
        from .free_placement import check_layout
        from .overlay import _polyline

        frame = self.frames.capture(CAMERA)
        if frame is None:
            return
        est = IdenticalCylinderEstimator()
        est.reset(self.task)
        found = est.estimate([frame])
        problems = check_layout(self.task, found, self.area)
        bad = {n for n in found if any(n in m for m in problems)}
        img = np.array(frame.rgb[..., :3], dtype=np.uint8, copy=True)
        x0, x1 = self.area.x
        y0, y1 = self.area.y
        _polyline(img, frame, np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]]), (40, 220, 60), 3)
        a = np.linspace(0, 2 * np.pi, 40)
        for r in self.task.scene.regions:
            c = np.asarray(r.center, float)
            if c[2] < 0.06:  # the goal's bottom layer, drawn on the table
                rad = 0.0254
                _polyline(img, frame, np.c_[c[0] + rad * np.cos(a), c[1] + rad * np.sin(a), np.zeros(40)], (255, 140, 0), 2)
        for n, e in found.items():
            if not e.found:
                continue
            o = self.task.scene.object(n)
            top = 2 * float(e.pos[2])
            col = (240, 40, 40) if n in bad else (40, 230, 60)
            _polyline(img, frame, np.c_[e.pos[0] + o.size[0] * np.cos(a), e.pos[1] + o.size[0] * np.sin(a),
                                        np.full(40, top)], col, 3)
        lines = [f"{sum(e.found for e in found.values())} of {len(found)} seen"] + (problems or ["layout OK: press Enter"])
        img = _caption(img, lines, ok=not problems)
        try:
            import imageio.v3 as iio
        except ImportError:  # pragma: no cover
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.png")
        iio.imwrite(tmp, img)
        tmp.replace(self.path)  # atomic: a viewer never reads half a file
        self.frames_written += 1


def _caption(img: np.ndarray, lines: list[str], ok: bool) -> np.ndarray:
    """Text on a dark band at the top of the image (Pillow if installed, else no text)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        return img
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=26)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover
        font = ImageFont.load_default()
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    h = 34 * len(lines) + 12
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((12, 6 + 34 * i), line, fill=(90, 240, 90) if ok else (255, 110, 110), font=font)
    return np.asarray(im)


def make_lab_env(task: TaskSpec, *, extrinsics: str | Path, port: int = 8765, speed: float = 0.1,
                 serial: str | None = None, picture: str | Path | None = "runs/real/placement.png",
                 free_placement: bool = False, predict=None, layout_dir: str | Path | None = None,
                 robot=None, frames=None, estimator=None, operator=None) -> RealEnv:
    """The lab cell as a `RealEnv`. `speed`: tool speed limit (m/s); 0.1 until 10 clean trials (T7 brief).

    free_placement: objects go anywhere in the start area (`FreePlacementOperator`); `predict`
    (pinned task -> Prediction) is the per-layout sim prediction, `layout_dir` where layouts are saved."""
    if estimator is None:
        if not identical_cylinders(task):
            raise ValueError(f"{task.id}: only tasks whose movable objects are identical cylinders are wired "
                             "for the lab camera so far (IdenticalCylinderEstimator)")
        estimator = IdenticalCylinderEstimator()
        estimator.reset(task)
    if frames is None:
        frames = RealSenseFrameSource([RealSenseCamera(CAMERA, load_extrinsics(extrinsics), serial=serial,
                                                       width=1280, height=720)])
    if operator is None:
        pic = PlacementPicture(task, frames, estimator, picture) if picture else None
        if free_placement:
            from .free_placement import FreePlacementOperator

            operator = FreePlacementOperator(task, predict=predict,
                                             fresh=lambda: estimator.reset(task),
                                             preview=Preview(task, frames, picture) if picture else None,
                                             out=Path(layout_dir) if layout_dir else None)
        else:
            operator = TerminalOperator(show=pic)
    if robot is None:
        robot = FrankaBridgeRobotIO(port=port)
    limits = SafetyLimits(ee_speed=speed)
    # report_contact_force=False: the Franka's force estimate is not a contact sensor (see RealEnv).
    return RealEnv(task, robot, frames, estimator, controller=MujocoController(task, max_ee_speed=speed),
                   operator=operator, limits=limits, cameras=(CAMERA,), realtime=True, report_contact_force=False)
