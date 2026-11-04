"""T4: evaluate a skill over many seeds on a process pool.

Skills cross the process boundary as `SkillSource` (code, not live objects); each worker
loads the skill and builds the env once per (task, skill) and reuses them across seeds,
exactly as serial `evaluate` reuses one env and one skill. Episodes are returned in seed
order, so the report equals the serial one for the same seeds (everything except the
wall-clock fields `Episode.duration_s` and `EvalReport.created_at`).

`make_env` must be picklable: a module-level function or class such as
`skillforge.sim.make_env` or `StubEnv`, not a lambda or closure.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from skillforge.contracts import Episode, EvalReport, SimEnv, Skill, SkillSource, TaskSpec

from .harness import run_episode, summarize

MakeEnv = Callable[[TaskSpec], SimEnv]

# ------------------------------------------------------------------ worker side

# Per-process cache: key -> (env, skill). One entry per (make_env, task, skill) seen.
_CACHE: dict[str, tuple[SimEnv, Skill]] = {}
_CACHE_MAX = 8


def _key(make_env: MakeEnv, src: SkillSource, task_json: str) -> str:
    h = hashlib.sha256()
    for part in (f"{make_env.__module__}.{getattr(make_env, '__qualname__', repr(make_env))}", src.model_dump_json(), task_json):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _get(make_env: MakeEnv, src: SkillSource, task_json: str) -> tuple[SimEnv, Skill, TaskSpec]:
    from skillforge.skills import load_skill  # imported here so workers pay for it only once

    task = TaskSpec.model_validate_json(task_json)
    k = _key(make_env, src, task_json)
    if k not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            old_env, _ = _CACHE.pop(next(iter(_CACHE)))
            old_env.close()
        _CACHE[k] = (make_env(task), load_skill(src))
    env, skill = _CACHE[k]
    return env, skill, task


def _run_chunk(make_env: MakeEnv, src: SkillSource, task_json: str, seeds: list[int]) -> list[str]:
    env, skill, task = _get(make_env, src, task_json)
    # Episodes go back as JSON: pydantic models pickle fine, but JSON keeps the wire format
    # identical to what lands on disk.
    return [run_episode(env, skill, task, s).model_dump_json() for s in seeds]


def _close_all() -> None:
    for env, _ in _CACHE.values():
        try:
            env.close()
        except Exception:
            pass
    _CACHE.clear()


# ------------------------------------------------------------------ parent side


def _chunks(seeds: Sequence[int], n_chunks: int) -> list[list[int]]:
    n_chunks = max(1, min(n_chunks, len(seeds)))
    size, rem = divmod(len(seeds), n_chunks)
    out, i = [], 0
    for c in range(n_chunks):
        j = i + size + (1 if c < rem else 0)
        out.append(list(seeds[i:j]))
        i = j
    return out


def _check_picklable(make_env: MakeEnv) -> None:
    try:
        pickle.dumps(make_env)
    except Exception as e:
        raise TypeError(
            f"make_env={make_env!r} cannot be sent to worker processes ({e}). "
            "Pass a module-level function or class (e.g. skillforge.sim.make_env), not a lambda/closure."
        ) from None


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1))


class EvalPool:
    """A reusable process pool for many evaluations (sweeps, regression suites).

    Use as a context manager. `workers <= 1` runs in-process with no pool at all.
    """

    def __init__(self, workers: int | None = None, mp_context: str = "spawn", chunks_per_worker: int = 4):
        self.workers = default_workers() if workers is None else workers
        self.chunks_per_worker = chunks_per_worker
        self._ctx = mp_context
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> EvalPool:
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context(self._ctx))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        _close_all()  # the in-process cache used when workers <= 1

    def episodes(self, make_env: MakeEnv, src: SkillSource, task: TaskSpec, seeds: Iterable[int]) -> list[Episode]:
        seeds = [int(s) for s in seeds]
        if not seeds:
            return []
        task_json = task.model_dump_json()
        if self._pool is None:
            if self.workers > 1:
                raise RuntimeError("EvalPool used outside its `with` block")
            raw = _run_chunk(make_env, src, task_json, seeds)
        else:
            _check_picklable(make_env)
            parts = _chunks(seeds, self.workers * self.chunks_per_worker)
            futs = [self._pool.submit(_run_chunk, make_env, src, task_json, p) for p in parts]
            raw = [ep for f in futs for ep in f.result()]  # submission order = seed order
        return [Episode.model_validate_json(r) for r in raw]

    def evaluate(
        self, make_env: MakeEnv, src: SkillSource, task: TaskSpec, seeds: Iterable[int], max_failures: int = 20
    ) -> tuple[EvalReport, list[Episode]]:
        eps = self.episodes(make_env, src, task, seeds)
        if eps:
            name, version = eps[0].skill, eps[0].skill_version
        else:  # no seeds: report under the source's identity
            name, version = src.name, src.version
        return summarize(task, name, version, eps, max_failures), eps


def evaluate_parallel(
    make_env: MakeEnv,
    skill_source: SkillSource,
    task: TaskSpec,
    seeds: Iterable[int],
    workers: int | None = None,
    max_failures: int = 20,
    mp_context: str = "spawn",
) -> tuple[EvalReport, list[Episode]]:
    """Parallel `evaluate`. Same report and episodes as serial for the same seeds."""
    with EvalPool(workers, mp_context=mp_context) as pool:
        return pool.evaluate(make_env, skill_source, task, seeds, max_failures)
