"""T4: record episodes to GIF through the env's `render` hook.

Recording replays a seed: episodes are deterministic per seed, so re-running seed s with
rendering on reproduces the evaluated episode. `record_episode` checks that the replay
matched (same success, same step count) and says so if it didn't.

Everything here degrades instead of failing: an env whose `render` returns None or raises,
or a missing `imageio`, gives `None` paths plus a reason, never an exception.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from skillforge.contracts import Episode, EvalReport, SimEnv, Skill, TaskSpec

from .harness import run_episode
from .parallel import MakeEnv

log = logging.getLogger(__name__)


class Recording(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    path: str | None
    n_frames: int
    reason: str = ""  # why there is no video, or a replay mismatch note
    episode: Episode | None = None


class _Unrenderable(Exception):
    pass


def _imageio():
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    return iio


def _to_frame(img: object) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise _Unrenderable(f"render returned shape {a.shape}, expected HxWx3")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a[..., :3]


def write_gif(frames: Sequence[np.ndarray], path: Path, fps: float = 10.0) -> Path:
    iio = _imageio()
    if iio is None:
        raise ImportError("imageio is not installed (pip install 'skillforge[video]')")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), extension=".gif", duration=int(round(1000 / fps)), loop=0)
    return path


def record_episode(
    env: SimEnv,
    skill: Skill,
    task: TaskSpec,
    seed: int,
    path: Path,
    camera: str | None = None,
    every: int = 2,
    max_frames: int = 150,
    fps: float = 10.0,
    expected: Episode | None = None,
) -> Recording:
    """Replay `seed` rendering every `every`-th step; write a GIF to `path` if any frames came back."""
    if _imageio() is None:
        return Recording(seed=seed, path=None, n_frames=0, reason="imageio not installed")
    frames: list[np.ndarray] = []
    state = {"why": ""}

    def hook(e: SimEnv, i: int) -> None:
        if state["why"] or i % every or len(frames) >= max_frames:
            return
        try:
            img = e.render(camera)
        except Exception as ex:  # e.g. no GL context on a headless box
            state["why"] = f"render raised {type(ex).__name__}: {ex}"
            return
        if img is None:
            state["why"] = "env cannot render (render returned None)"
            return
        try:
            frames.append(_to_frame(img))
        except _Unrenderable as ex:
            state["why"] = str(ex)

    ep = run_episode(env, skill, task, seed, on_step=hook)
    note = ""
    if expected is not None and (ep.success, ep.n_steps) != (expected.success, expected.n_steps):
        note = (
            f"replay mismatch: eval had success={expected.success} n_steps={expected.n_steps}, "
            f"replay success={ep.success} n_steps={ep.n_steps} (env or skill not deterministic per seed)"
        )
        log.warning(note)
    if not frames:
        return Recording(seed=seed, path=None, n_frames=0, reason=state["why"] or "no frames", episode=ep)
    out = write_gif(frames, path, fps)
    ep = ep.model_copy(update={"video": str(out)})
    return Recording(seed=seed, path=str(out), n_frames=len(frames), reason=note or state["why"], episode=ep)


def select_episodes(eps: Sequence[Episode], n_success: int = 1) -> list[Episode]:
    """The first failure of each tag (in seed order), plus the first `n_success` successes."""
    seen: set[str] = set()
    out = []
    for e in eps:
        if not e.success and e.failure not in seen:
            seen.add(e.failure)
            out.append(e)
    out += [e for e in eps if e.success][:n_success]
    return out


def record_selected(
    make_env: MakeEnv,
    skill: Skill,
    task: TaskSpec,
    eps: Sequence[Episode],
    out_dir: Path,
    report: EvalReport | None = None,
    **kw: object,
) -> tuple[list[Recording], list[Episode], EvalReport | None]:
    """Record `select_episodes(eps)`; return recordings, `eps` with `video` filled in, and `report`
    with `FailureCase.video` filled in. Stops after the first episode that cannot render."""
    out_dir = Path(out_dir)
    env = make_env(task)
    recs: list[Recording] = []
    try:
        for e in select_episodes(eps):
            label = "success" if e.success else e.failure
            r = record_episode(env, skill, task, e.seed, out_dir / f"{task.id}_{skill.version}_s{e.seed}_{label}.gif", expected=e, **kw)
            recs.append(r)
            if r.path is None:
                log.info("no video for %s: %s", task.id, r.reason)
                break  # rendering unavailable: don't replay the rest for nothing
    finally:
        env.close()
    vids = {r.seed: r.path for r in recs if r.path}
    new_eps = [e.model_copy(update={"video": vids[e.seed]}) if e.seed in vids else e for e in eps]
    if report is not None:
        report = report.model_copy(
            update={"failures": [f.model_copy(update={"video": vids.get(f.seed, f.video)}) for f in report.failures]}
        )
    return recs, new_eps, report
