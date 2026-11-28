"""Fetch robot models at runtime into the git-ignored `assets/cache/`. Never commit meshes.

    python -m skillforge.sim.assets            # fetch the pinned menagerie robots
    SKILLFORGE_ASSETS=/some/dir python -m ...  # use another cache directory
    SKILLFORGE_NO_FETCH=1                      # never touch the network (tests then skip)

The menagerie is fetched with a shallow, blob-filtered, sparse git checkout pinned to
`MENAGERIE_COMMIT`, so only the robot directories we use land on disk (~45 MB for the Panda).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"  # pinned for reproducible physics
DEFAULT_ROBOTS = ("franka_emika_panda",)


class AssetError(RuntimeError):
    """The model files are missing and could not be fetched (offline, no git, ...)."""


def cache_dir() -> Path:
    env = os.environ.get("SKILLFORGE_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    # src/skillforge/sim/assets.py -> repo root (editable install), else the working directory
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").exists():
        root = Path.cwd()
    return root / "assets" / "cache"


def menagerie_dir() -> Path:
    return cache_dir() / "mujoco_menagerie"


def _git(args: list[str], cwd: Path) -> None:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise AssetError(f"git {' '.join(args)} failed: {r.stderr.strip()[-500:]}")


def fetch_menagerie(robots: tuple[str, ...] = DEFAULT_ROBOTS) -> Path:
    """Make sure each robot dir exists under the menagerie cache; fetch what's missing."""
    dest = menagerie_dir()
    missing = [r for r in robots if not (dest / r).is_dir()]
    if not missing:
        return dest
    if shutil.which("git") is None:
        raise AssetError("git is not installed; cannot fetch mujoco_menagerie")
    if (dest / ".git").is_dir():  # existing sparse checkout: widen it
        have = [p.name for p in dest.iterdir() if p.is_dir() and not p.name.startswith(".")]
        _git(["sparse-checkout", "set", *sorted(set(have) | set(robots))], dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="menagerie-", dir=dest.parent))
    try:
        _git(["init", "-q"], tmp)
        _git(["remote", "add", "origin", MENAGERIE_URL], tmp)
        _git(["sparse-checkout", "set", *robots], tmp)
        _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT], tmp)
        _git(["checkout", "-q", "FETCH_HEAD"], tmp)
        if dest.exists():  # another process won the race
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


# Default model file per menagerie dir, for `menagerie:<dir>` sources without an explicit file.
MENAGERIE_MODEL_FILES = {"franka_emika_panda": "panda.xml", "kinova_gen3": "gen3.xml", "robotiq_2f85": "2f85.xml"}


def resolve_robot_source(source: str, fetch: bool | None = None) -> Path:
    """Turn a RobotSpec.source into a model file path, fetching menagerie assets if allowed.

    Fetching is on by default; SKILLFORGE_NO_FETCH=1 turns it off (e.g. offline CI).
    """
    if fetch is None:
        fetch = os.environ.get("SKILLFORGE_NO_FETCH") != "1"
    kind, _, rest = source.partition(":")
    if kind == "menagerie":
        robot, _, fname = rest.partition("/")
        fname = fname or MENAGERIE_MODEL_FILES.get(robot, "scene.xml")
        path = menagerie_dir() / robot / fname
        if not path.exists() and fetch:
            fetch_menagerie((robot,))
        if not path.exists():
            raise AssetError(f"model file {path} not found (run `python -m skillforge.sim.assets`)")
        return path
    if kind in ("mjcf", "urdf"):
        path = Path(rest).expanduser()
        if not path.exists():
            raise AssetError(f"model file {path} not found")
        return path
    raise ValueError(f"unknown robot source '{source}'")


if __name__ == "__main__":
    print(fetch_menagerie())
