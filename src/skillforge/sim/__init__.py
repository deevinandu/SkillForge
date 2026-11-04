"""T1: simulation backends. Every env implements contracts.SimEnv."""

import os
import sys


def default_mujoco_gl() -> None:
    """Pick a headless GL backend if none is set. MuJoCo reads MUJOCO_GL once, at import.

    CPU-only Linux box: osmesa (needs libosmesa6). GPU box: export MUJOCO_GL=egl yourself.
    """
    if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "osmesa"


default_mujoco_gl()

from .stub_env import StubEnv  # noqa: E402
from .success import is_success  # noqa: E402

PHYSICS_SOURCES = ("menagerie:", "mjcf:", "urdf:")


def make_env(task, **kwargs):
    """Pick a backend from task.scene.robot.source.

    'stub' -> StubEnv; 'menagerie:<dir>[/<file>]', 'mjcf:<path>', 'urdf:<path>' -> MujocoEnv
    (needs the `sim` extra; menagerie models are fetched into assets/cache/ on first use).
    """
    src = task.scene.robot.source
    if src == "stub":
        return StubEnv(task)
    if src.startswith(PHYSICS_SOURCES):
        from .mujoco_env import MujocoEnv  # lazy: mujoco is an optional dependency

        return MujocoEnv(task, **kwargs)
    raise NotImplementedError(f"no sim backend for robot source '{src}'")


__all__ = ["StubEnv", "default_mujoco_gl", "is_success", "make_env"]
