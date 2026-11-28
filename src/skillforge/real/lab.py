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

            operator = FreePlacementOperator(task, predict=predict, show=(lambda: pic(0, {})) if pic else None,
                                             out=Path(layout_dir) if layout_dir else None)
        else:
            operator = TerminalOperator(show=pic)
    if robot is None:
        robot = FrankaBridgeRobotIO(port=port)
    limits = SafetyLimits(ee_speed=speed)
    return RealEnv(task, robot, frames, estimator, controller=MujocoController(task, max_ee_speed=speed),
                   operator=operator, limits=limits, cameras=(CAMERA,), realtime=True)
