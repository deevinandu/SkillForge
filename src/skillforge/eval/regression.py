"""T4: regression suite. Pinned seeds, stored baselines, fail when a skill gets worse.

A `Baseline` records how one skill version did on one task over a pinned seed list. A later
version of that skill passes `check` when its success rate on the SAME seeds is at least the
baseline's Wilson 95% lower bound (or an explicit `min_success`, whichever is higher). The
baseline also stores the task's hash, so a baseline taken on a different task definition is
reported as stale rather than silently compared.

Baselines live as JSON, one per (task, skill): `<dir>/<task_id>__<skill>.json`. The pytest
entry point is `tests/test_regression_suite.py` (`pytest -m regression`); regenerate with
`python -m skillforge.eval regression update`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from skillforge.contracts import EvalReport, Skill, SkillSource, TaskSpec

from .parallel import EvalPool, MakeEnv

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(100))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def task_hash(task: TaskSpec) -> str:
    return sha(task.model_dump_json())


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_sha: str
    skill: str
    skill_version: str
    code_sha: str
    seeds: list[int]
    n_success: int
    success_rate: float
    ci95: tuple[float, float]
    failure_counts: dict[str, int]
    passed_seeds: list[int]
    min_success: float | None = None  # optional absolute floor on top of the CI bound
    created_at: str

    @property
    def threshold(self) -> float:
        return max(self.ci95[0], self.min_success or 0.0)


class RegressionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill: str
    baseline_version: str
    candidate_version: str
    baseline_rate: float
    candidate_rate: float
    threshold: float
    passed: bool
    stale: bool  # the task changed since the baseline was taken
    newly_failing: list[int]  # seeds the baseline passed and the candidate fails
    newly_passing: list[int]
    candidate: EvalReport
    reason: str

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.skill} {self.baseline_version} -> {self.candidate_version} on {self.task_id}: "
            f"{self.candidate_rate:.0%} vs baseline {self.baseline_rate:.0%} (threshold {self.threshold:.0%}); "
            f"{self.reason}"
        )


# ------------------------------------------------------------------ baseline and check


def make_baseline(
    make_env: MakeEnv,
    src: SkillSource,
    task: TaskSpec,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    pool: EvalPool | None = None,
    min_success: float | None = None,
) -> Baseline:
    seeds = [int(s) for s in seeds]
    with _maybe_pool(pool) as p:
        report, eps = p.evaluate(make_env, src, task, seeds)
    return Baseline(
        task_id=task.id,
        task_sha=task_hash(task),
        skill=src.name,
        skill_version=src.version,
        code_sha=sha(src.code),
        seeds=seeds,
        n_success=report.n_success,
        success_rate=report.success_rate,
        ci95=report.ci95,
        failure_counts=report.failure_counts,
        passed_seeds=[e.seed for e in eps if e.success],
        min_success=min_success,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def check(
    make_env: MakeEnv,
    candidate: SkillSource,
    task: TaskSpec,
    baseline: Baseline,
    pool: EvalPool | None = None,
) -> RegressionResult:
    """Run `candidate` on the baseline's pinned seeds and compare."""
    if baseline.task_id != task.id:
        raise ValueError(f"baseline is for task '{baseline.task_id}', not '{task.id}'")
    with _maybe_pool(pool) as p:
        report, eps = p.evaluate(make_env, candidate, task, baseline.seeds)
    stale = baseline.task_sha != task_hash(task)
    was = set(baseline.passed_seeds)
    now = {e.seed for e in eps if e.success}
    ok = report.success_rate >= baseline.threshold
    if stale:
        reason = "task definition changed since the baseline; re-baseline before trusting this"
    elif ok:
        reason = "at or above threshold"
    else:
        top = sorted(report.failure_counts.items(), key=lambda kv: -kv[1])[:3]
        reason = f"below threshold; top failures: {dict(top)}"
    return RegressionResult(
        task_id=task.id,
        skill=candidate.name,
        baseline_version=baseline.skill_version,
        candidate_version=candidate.version,
        baseline_rate=baseline.success_rate,
        candidate_rate=report.success_rate,
        threshold=baseline.threshold,
        passed=ok and not stale,
        stale=stale,
        newly_failing=sorted(was - now),
        newly_passing=sorted(now - was),
        candidate=report,
        reason=reason,
    )


def compare(
    make_env: MakeEnv,
    candidate: SkillSource,
    baseline_src: SkillSource,
    task: TaskSpec,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    pool: EvalPool | None = None,
) -> RegressionResult:
    """Compare two skill versions head to head on the same seeds (no stored baseline needed)."""
    seeds = list(seeds)
    with _maybe_pool(pool) as p:
        base = make_baseline(make_env, baseline_src, task, seeds, pool=p)
        return check(make_env, candidate, task, base, pool=p)


class _maybe_pool:
    """Use the caller's open pool, or open (and close) a serial one."""

    def __init__(self, pool: EvalPool | None):
        self.pool, self.own = pool, pool is None

    def __enter__(self) -> EvalPool:
        if self.own:
            self.pool = EvalPool(workers=1).__enter__()
        return self.pool

    def __exit__(self, *exc: object) -> None:
        if self.own:
            self.pool.close()


# ------------------------------------------------------------------ storage and discovery


def baseline_path(directory: Path, task_id: str, skill: str) -> Path:
    return Path(directory) / f"{task_id}__{skill}.json"


def save_baseline(b: Baseline, directory: Path) -> Path:
    p = baseline_path(directory, b.task_id, b.skill)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(b.model_dump_json(indent=2) + "\n")
    return p


def load_baselines(directory: Path) -> list[Baseline]:
    return [Baseline.model_validate_json(p.read_text()) for p in sorted(Path(directory).glob("*__*.json"))]


def load_tasks(directory: Path) -> dict[str, TaskSpec]:
    tasks = {}
    for p in sorted(Path(directory).glob("*.json")):
        t = TaskSpec.model_validate(json.loads(p.read_text()))
        tasks[t.id] = t
    return tasks


def source_of(cls: type) -> SkillSource:
    """A SkillSource for a hand-written skill class: its whole module as code."""
    mod = inspect.getmodule(cls)
    return SkillSource(
        name=cls.name, version=cls.version, code=inspect.getsource(mod), entrypoint=cls.__name__
    )


def _version_key(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in v.split("."))


def library_sources() -> dict[str, SkillSource]:
    """Skills currently exported by `skillforge.skills`, keyed by skill name.

    Until T2/T3 add an accepted-skill library this is the hand-written skills package.
    """
    import skillforge.skills as lib

    out = {}
    for name in getattr(lib, "__all__", []):
        obj = getattr(lib, name)
        if isinstance(obj, type) and {"name", "version", "reset", "act"} <= set(dir(obj)):
            try:
                inst = obj()
            except Exception:
                continue
            if isinstance(inst, Skill):
                src = source_of(obj)
                prev = out.get(src.name)
                if prev is None or _version_key(src.version) > _version_key(prev.version):
                    out[src.name] = src  # several classes may share a name; keep the newest
    return out


def update_baselines(
    make_env: MakeEnv,
    tasks: dict[str, TaskSpec],
    sources: dict[str, SkillSource],
    directory: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    workers: int | None = None,
) -> list[Path]:
    """(Re)write a baseline for every task x skill pair where the skill applies.

    A pair "applies" when the skill succeeds at least once; a skill built for another
    kind of task (e.g. PickPlace on a lift task) would only record a 0% baseline.
    Tasks whose env backend is not available yet (NotImplementedError) are skipped.
    """
    written = []
    with EvalPool(workers) as pool:
        for task in tasks.values():
            for src in sources.values():
                try:
                    b = make_baseline(make_env, src, task, seeds, pool=pool)
                except NotImplementedError:
                    break
                if b.n_success > 0:
                    written.append(save_baseline(b, directory))
    return written
