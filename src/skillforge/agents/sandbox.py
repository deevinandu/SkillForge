"""Run agent-written skill code in a subprocess, never in the forge process.

What the sandbox does:
- a fresh `python -I` interpreter (no user site, no cwd on sys.path) in a throwaway temp dir,
- a minimal environment: API keys and other secrets from the parent are NOT passed through,
- a wall-clock timeout; on expiry the whole process group is killed,
- best-effort network denial (socket creation and DNS raise inside the child),
- RLIMIT_AS memory cap and no core dumps on POSIX.

What it does NOT do (yet): filesystem isolation, OS-level network namespaces, seccomp. Agent code
can still read files the user can read. Run on a disposable box if that matters.

A sandbox failure (load error, crash, timeout) becomes a synthetic EvalReport with every rollout
tagged 'exception' and the reason in the failure detail, so the agent can read what went wrong.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from skillforge.contracts import EvalReport, FailureCase, SimEnv, SkillSource, TaskSpec
from skillforge.eval import wilson

DEFAULT_MAX_MEMORY = 4 * 1024**3
_PASS_ENV = ("PATH", "LANG", "LC_ALL", "MUJOCO_GL", "PYOPENGL_PLATFORM", "SYSTEMROOT", "TMPDIR")


class SandboxError(RuntimeError):
    pass


@dataclass
class SandboxResult:
    ok: bool
    stage: str  # 'check' | 'eval' | 'load' | 'smoke' | 'timeout' | 'crash'
    error: str = ""
    report: EvalReport | None = None
    wall_s: float = 0.0
    outcomes: dict[int, bool] | None = None  # eval only: seed -> success, every seed (not a sample)


def env_ref(make_env: Callable[[TaskSpec], SimEnv] | str) -> str:
    """'module:qualname' for an importable env factory, so the child can rebuild it."""
    if isinstance(make_env, str):
        return make_env
    mod, qual = getattr(make_env, "__module__", None), getattr(make_env, "__qualname__", "")
    if not mod or "<" in qual:
        raise SandboxError(
            f"make_env {make_env!r} is not importable by reference (lambda/local?); pass a "
            "module-level function or class, or a 'module:attr' string"
        )
    import importlib

    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part, None)
    if obj is not make_env:
        raise SandboxError(f"make_env {mod}:{qual} does not resolve to the same object in a fresh process")
    return f"{mod}:{qual}"


def _child_env(tmp: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASS_ENV if k in os.environ}
    env.update(HOME=tmp, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", NO_PROXY="*")
    return env


def _run(req: dict, timeout_s: float) -> SandboxResult:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="skillforge-sbx-") as tmp:
        req_path, out_path = Path(tmp, "request.json"), Path(tmp, "result.json")
        req_path.write_text(json.dumps(req))
        cmd = [sys.executable, "-I", "-m", "skillforge.agents._sandbox_worker", str(req_path), str(out_path)]
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            env=_child_env(tmp),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill(proc)
            return SandboxResult(False, "timeout", f"sandbox timeout after {timeout_s:.1f}s", wall_s=time.perf_counter() - t0)
        wall = time.perf_counter() - t0
        if not out_path.exists():
            tail = err.decode(errors="replace")[-3000:]
            return SandboxResult(False, "crash", f"sandbox exited {proc.returncode} without a result:\n{tail}", wall_s=wall)
        res = json.loads(out_path.read_text())
    report = EvalReport.model_validate(res["report"]) if res.get("report") else None
    outcomes = {int(s): bool(ok) for s, ok in res["outcomes"]} if res.get("outcomes") is not None else None
    return SandboxResult(res["ok"], res["stage"], res.get("error", ""), report, wall, outcomes)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()
    proc.communicate()


def _request(mode: str, task: TaskSpec, src: SkillSource, make_env, max_memory: int | None, **extra) -> dict:
    return {
        "mode": mode,
        "task": json.loads(task.model_dump_json()),
        "source": json.loads(src.model_dump_json()),
        "env_ref": env_ref(make_env),
        "max_memory_bytes": max_memory,
        **extra,
    }


def check_source(
    task: TaskSpec,
    src: SkillSource,
    make_env: Callable[[TaskSpec], SimEnv] | str,
    timeout_s: float = 30.0,
    smoke_steps: int = 5,
    max_memory: int | None = DEFAULT_MAX_MEMORY,
) -> SandboxResult:
    """Compile, instantiate, protocol-check and smoke-step the skill in a subprocess."""
    return _run(_request("check", task, src, make_env, max_memory, smoke_steps=smoke_steps), timeout_s)


def evaluate_source(
    task: TaskSpec,
    src: SkillSource,
    make_env: Callable[[TaskSpec], SimEnv] | str,
    seeds: Iterable[int],
    timeout_s: float = 600.0,
    max_failures: int = 20,
    max_memory: int | None = DEFAULT_MAX_MEMORY,
) -> SandboxResult:
    """Run T4's `evaluate` on the skill in a subprocess. `result.report` is always set, and so is
    `result.outcomes` (seed -> success; all False when the sandbox failed)."""
    seeds = list(seeds)
    req = _request("eval", task, src, make_env, max_memory, seeds=seeds, max_failures=max_failures)
    res = _run(req, timeout_s)
    if res.report is None:
        res.report = failure_report(task, src, seeds, f"sandbox {res.stage}: {res.error}")
    if res.outcomes is None or not res.ok:
        res.outcomes = {s: False for s in seeds}
    return res


def failure_report(task: TaskSpec, src: SkillSource, seeds: list[int], detail: str) -> EvalReport:
    n = len(seeds)
    return EvalReport(
        task_id=task.id,
        skill=src.name,
        skill_version=src.version,
        n_rollouts=n,
        n_success=0,
        success_rate=0.0,
        ci95=wilson(0, n),
        failure_counts={"exception": n} if n else {},
        failures=[FailureCase(seed=seeds[0] if seeds else -1, tag="exception", detail=detail[-4000:], randomization={})],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
