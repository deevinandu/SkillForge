"""`python -m skillforge.perception <command>`. Output goes under runs/ (gitignored) by default.

  compare  success of one skill with the env's own poses ("true": true pose plus the task's
           obs noise), the oracle wrapper and depth perception, on the same seeds, with paired counts
  error    estimator error against true sim poses -> perception.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "tasks"
RUNS = REPO / "runs"


def _task(path_or_id: str):
    from skillforge.contracts import TaskSpec

    p = Path(path_or_id)
    if not p.is_file():
        p = TASKS / f"{path_or_id}.json"
    return TaskSpec.model_validate(json.loads(p.read_text()))


def _source(spec: str | None):
    """None -> the library PickPlace; else 'path/to/skill.py[:Class]' (Class defaults to PickPlace)."""
    from skillforge.contracts import SkillSource

    if spec is None:
        from skillforge.eval.regression import source_of
        from skillforge.skills import PickPlace

        return source_of(PickPlace)
    path, _, cls = spec.partition(":")
    p = Path(path)
    return SkillSource(name=p.stem, version="file", code=p.read_text(), entrypoint=cls or "PickPlace")


def _seeds(spec: str) -> range:
    lo, _, hi = spec.partition("-")
    return range(int(lo), int(hi))


def cmd_compare(a: argparse.Namespace) -> None:
    # Spawned workers inherit this. Without it each worker's BLAS starts a thread per core and
    # they fight: measured 2.4x slower on 12 workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from skillforge.agents.library import paired_compare
    from skillforge.eval.parallel import evaluate_parallel
    from skillforge.sim import make_env

    from .env import make_depth_env, make_noisy_depth_env, make_oracle_env

    task, src, seeds = _task(a.task), _source(a.skill), _seeds(a.seeds)
    outcomes, reports = {}, {}
    envs = [("true", make_env), ("oracle", make_oracle_env), ("depth", make_depth_env)]
    if a.noisy:
        envs.append(("noisy", make_noisy_depth_env))
    for label, mk in envs:
        rep, eps = evaluate_parallel(mk, src, task, seeds, workers=a.workers)
        outcomes[label] = {e.seed: e.success for e in eps}
        reports[label] = json.loads(rep.model_dump_json())
        print(f"{label:6s} {rep.summary()}")
    pairs = {}
    for ref, cand in [("true", label) for label, _ in envs[1:]]:
        c = paired_compare(outcomes[ref], outcomes[cand])
        pairs[f"{ref}_vs_{cand}"] = {"both": c.both, f"{ref}_only": c.best_only, f"{cand}_only": c.cand_only, "neither": c.neither}
        print(f"{ref} vs {cand}: both {c.both}, {ref} only {c.best_only}, {cand} only {c.cand_only}, neither {c.neither}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": task.id, "seeds": [seeds.start, seeds.stop], "reports": reports, "paired": pairs}, indent=2))
    print(f"wrote {out}")


def cmd_error(a: argparse.Namespace) -> None:
    from skillforge.skills import load_skill

    from .env import make_depth_env
    from .error_model import measure

    src = _source(a.skill)
    skill = load_skill(src)  # trusted code only: this runs in-process
    rep = measure(make_depth_env, skill, _task(a.task), _seeds(a.seeds))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.model_dump_json(indent=2))
    print(rep.summary())
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skillforge.perception", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, default_out in (
        ("compare", cmd_compare, RUNS / "perception" / "compare.json"),
        ("error", cmd_error, RUNS / "perception" / "perception.json"),
    ):
        p = sub.add_parser(name)
        p.add_argument("task", help="task id or path to a TaskSpec json")
        p.add_argument("--skill", default=None, help="path/to/skill.py[:Class]; default: library PickPlace")
        p.add_argument("--seeds", default="5000-5200", help="lo-hi, hi exclusive")
        p.add_argument("--out", default=str(default_out))
        if name == "compare":
            p.add_argument("--workers", type=int, default=None)
            p.add_argument("--noisy", action="store_true", help="also run depth perception with depth noise")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0
