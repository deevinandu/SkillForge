"""`python -m skillforge.perception <command>`. Output goes under runs/ (gitignored) by default.

  compare  success of one skill with the env's own poses ("true": true pose plus the task's
           obs noise), the oracle wrapper and depth perception, on the same seeds, with paired counts
           --envs picks the settings: true, oracle, depth, noisy (depth noise), color (colour
           segmentation), dr (visual randomization), dr-color (colour + noise + randomization)
  calibration  success against camera calibration error alone, at several rotation errors
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


def _threads() -> None:
    # Spawned workers inherit this. Without it each worker's BLAS starts a thread per core and
    # they fight: measured 2.4x slower on 12 workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _envs() -> dict:
    from skillforge.sim import make_env

    from . import env

    return {
        "true": make_env,  # the env's own poses: true pose plus the task's obs noise
        "oracle": env.make_oracle_env,
        "depth": env.make_depth_env,
        "noisy": env.make_noisy_depth_env,
        "color": env.make_color_depth_env,
        "dr": env.make_dr_depth_env,
        "dr-color": env.make_dr_color_depth_env,
    }


def _run(envs: list[tuple[str, object]], a: argparse.Namespace, extra: dict | None = None) -> None:
    """Evaluate one skill in each env on the same seeds; print and save reports and paired counts."""
    from skillforge.agents.library import paired_compare
    from skillforge.eval.parallel import evaluate_parallel

    _threads()
    task, src, seeds = _task(a.task), _source(a.skill), _seeds(a.seeds)
    outcomes, reports = {}, {}
    width = max(len(label) for label, _ in envs)
    for label, mk in envs:
        rep, eps = evaluate_parallel(mk, src, task, seeds, workers=a.workers)
        outcomes[label] = {e.seed: e.success for e in eps}
        reports[label] = json.loads(rep.model_dump_json())
        print(f"{label:{width}s} {rep.summary()}", flush=True)
    pairs = {}
    ref = envs[0][0]
    for cand, _ in envs[1:]:
        c = paired_compare(outcomes[ref], outcomes[cand])
        pairs[f"{ref}_vs_{cand}"] = {"both": c.both, f"{ref}_only": c.best_only, f"{cand}_only": c.cand_only, "neither": c.neither}
        print(f"{ref} vs {cand}: both {c.both}, {ref} only {c.best_only}, {cand} only {c.cand_only}, neither {c.neither}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"task": task.id, "seeds": [seeds.start, seeds.stop], "reports": reports, "paired": pairs, **(extra or {})}
    out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {out}")


def cmd_compare(a: argparse.Namespace) -> None:
    table = _envs()
    names = [n.strip() for n in a.envs.split(",")] + (["noisy"] if a.noisy else [])
    unknown = [n for n in names if n not in table]
    if unknown:
        raise SystemExit(f"unknown env(s) {unknown}; have {list(table)}")
    _run([(n, table[n]) for n in dict.fromkeys(names)], a)


def cmd_calibration(a: argparse.Namespace) -> None:
    """Success against camera calibration error alone (sim masks, clean depth, nothing else randomized)."""
    from functools import partial

    from .env import make_depth_env, make_perceived_env
    from .frames import VisualRandomization

    envs = [("0 deg", make_depth_env)]
    for deg in (float(x) for x in a.rot_deg.split(",")):
        visual = VisualRandomization.calibration_only(a.trans_mm / 1e3, deg)
        envs.append((f"{deg:g} deg", partial(make_perceived_env, visual=visual)))
    _run(envs, a, extra={"trans_mm": a.trans_mm})


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
        ("calibration", cmd_calibration, RUNS / "perception" / "calibration.json"),
        ("error", cmd_error, RUNS / "perception" / "perception.json"),
    ):
        p = sub.add_parser(name)
        p.add_argument("task", help="task id or path to a TaskSpec json")
        p.add_argument("--skill", default=None, help="path/to/skill.py[:Class]; default: library PickPlace")
        p.add_argument("--seeds", default="5000-5200", help="lo-hi, hi exclusive")
        p.add_argument("--out", default=str(default_out))
        if name != "error":
            p.add_argument("--workers", type=int, default=None)
        if name == "compare":
            p.add_argument("--envs", default="true,oracle,depth", help="comma list; the first is the reference. "
                           "true, oracle, depth, noisy, color, dr, dr-color")
            p.add_argument("--noisy", action="store_true", help="same as adding 'noisy' to --envs")
        if name == "calibration":
            p.add_argument("--rot-deg", default="0.25,0.5,1.0", help="max rotation error per camera, degrees")
            p.add_argument("--trans-mm", type=float, default=3.0, help="max translation error per axis, mm")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0
