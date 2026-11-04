"""T1: simulation backends. Every env implements contracts.SimEnv."""

from .stub_env import StubEnv
from .success import is_success


def make_env(task):
    """Pick a backend from task.scene.robot.source. T1 adds 'menagerie:' / 'mjcf:' / 'urdf:' here."""
    src = task.scene.robot.source
    if src == "stub":
        return StubEnv(task)
    raise NotImplementedError(f"no sim backend for robot source '{src}' yet (T1)")


__all__ = ["StubEnv", "is_success", "make_env"]
