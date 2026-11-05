"""`python -m skillforge.learning <command>`: the T5 pipeline. Everything lands under runs/.

  collect   roll the teacher out over seeds into a dataset (optionally with DART noise)
  train     behavior-clone a policy from one or more datasets
  dagger    rounds of: roll the learner out, label with the shadow teacher, aggregate, retrain
  eval      teacher vs learned at nominal and stressed randomization, held-out seeds, Wilson CIs
            (alias: compare)

Seed plan (disjoint on purpose): teacher data 0.., DAgger round r at 100000*r.., evaluation at
1000000.. by default. Typical run on a 4-CPU box::

  python -m skillforge.learning collect --name franka_v02 --seeds 1000
  python -m skillforge.learning collect --name franka_v02_dart --seeds 1000 --seed-start 50000 --dart 0.015
  python -m skillforge.learning train --data franka_v02 --out runs/policies/bc_plain/policy.pt
  python -m skillforge.learning train --data franka_v02 franka_v02_dart --out runs/policies/bc_dart/policy.pt
  cp -r runs/datasets/franka_v02 runs/datasets/franka_v02_dagger   # DAgger aggregates into a copy
  python -m skillforge.learning dagger --data franka_v02_dagger --policy runs/policies/bc_plain/policy.pt --rounds 2 --seeds 300
  python -m skillforge.learning eval --policy runs/policies/bc_plain/policy.pt --seeds 500 --seed-start 2000000

`eval` caches the teacher's results under runs/compare_cache/ (keyed on teacher code, task,
seeds and the sim backend's source), so comparing several policies re-runs only the learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from skillforge.contracts import TaskSpec

from .dataset import (
    DEFAULT_ENV,
    DEFAULT_TEACHER,
    CollectConfig,
    DartNoise,
    Dataset,
    collect_to,
    default_dataset_dir,
    resolve_spec,
)

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"
EVAL_SEED_START = 1_000_000
DAGGER_SEED_STRIDE = 100_000


def load_task(path_or_id: str) -> TaskSpec:
    p = Path(path_or_id)
    if not p.is_file():
        p = REPO / "tasks" / f"{path_or_id}.json"
    return TaskSpec.model_validate(json.loads(p.read_text()))


def stressed(task: TaskSpec, factor: float | None) -> TaskSpec:
    """`task` under T4's stress randomization (`stress_spec` x factor); unchanged for None/0."""
    if not factor:
        return task
    from skillforge.eval.robustness import stress_spec, with_randomization

    return with_randomization(task, stress_spec(task.scene.randomization, factor))


def _ds_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_dir() or "/" in name else default_dataset_dir(name)


def _progress(label: str):
    t0 = time.perf_counter()
    last = [0.0]

    def cb(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 10 or done == total:
            last[0] = now
            print(f"  {label}: {done}/{total} episodes, {now - t0:.0f}s", flush=True)

    return cb


# ---------------------------------------------------------------- commands


def cmd_collect(a: argparse.Namespace) -> dict:
    task = stressed(load_task(a.task), a.collect_stress)
    dart = DartNoise(pos_std=a.dart, p=a.dart_p) if a.dart else None
    cfg = CollectConfig(
        source="dart" if dart else "teacher",
        teacher=a.teacher,
        make_env=a.make_env,
        dart=dart,
        success_only=a.success_only,
    )
    seeds = range(a.seed_start, a.seed_start + a.seeds)
    ds, summ = collect_to(_ds_path(a.name), task, seeds, cfg, workers=a.workers, overwrite=a.overwrite,
                          progress=_progress("collect"))
    print(json.dumps({"dataset": str(ds.root), **summ, "total_episodes": ds.info["total_episodes"],
                      "total_frames": ds.info["total_frames"]}))
    return summ


def _train_cfg(a: argparse.Namespace):
    from .policy import TrainConfig

    return TrainConfig(
        n_hist=a.n_hist, horizon=a.horizon, hidden=tuple(a.hidden), epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, seed=a.train_seed, threads=a.threads, max_minutes=a.max_minutes,
    )


def cmd_train(a: argparse.Namespace) -> dict:
    from .policy import train

    dss = [Dataset(_ds_path(n)) for n in a.data]
    meta = train(dss, a.out, _train_cfg(a))
    print(json.dumps({k: meta[k] for k in ("n_frames", "n_episodes", "epochs_run", "train_seconds")} | {"checkpoint": a.out}))
    return meta


def cmd_dagger(a: argparse.Namespace) -> list[dict]:
    from .policy import train

    task = stressed(load_task(a.task), a.collect_stress)
    root = _ds_path(a.data)
    extra = [Dataset(_ds_path(n)) for n in a.extra_data]
    policy = Path(a.policy)
    out_dir = Path(a.out_dir) if a.out_dir else policy.parent.parent
    log = []
    for r in range(a.start_round, a.start_round + a.rounds):
        cfg = CollectConfig(source=f"dagger:r{r}", teacher=a.teacher, make_env=a.make_env,
                            learner_checkpoint=str(policy.resolve()), beta=a.beta,
                            learner_overrides={"ensemble_m": a.ensemble_m}, relabel=a.relabel,
                            sync_tol=None if a.relabel == "phase_sync" and a.sync_tol == "auto" else
                            (0.03 if a.sync_tol == "auto" else a.sync_tol),
                            sync_patience=a.sync_patience)
        start = DAGGER_SEED_STRIDE * r + a.seed_offset
        seeds = range(start, start + a.seeds)
        _, summ = collect_to(root, task, seeds, cfg, workers=a.workers, progress=_progress(f"dagger r{r}"))
        print(f"round {r}: learner {summ['success']}/{summ['rolled']} on its own rollouts; "
              f"{summ['frames_rolled']} frames relabeled in {summ['seconds']}s", flush=True)
        policy = out_dir / f"{a.prefix}_r{r}" / "policy.pt"
        meta = train([Dataset(root), *extra], policy, _train_cfg(a))
        log.append({"round": r, "rollouts": summ, "checkpoint": str(policy), "train_seconds": meta["train_seconds"],
                    "n_frames": meta["n_frames"]})
        print(json.dumps(log[-1]), flush=True)
    return log


def _sim_fingerprint() -> str:
    import skillforge.sim

    h = hashlib.sha256()
    for p in sorted(Path(skillforge.sim.__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    try:
        import mujoco

        h.update(mujoco.__version__.encode())
    except ImportError:
        pass
    return h.hexdigest()


def compare(
    task: TaskSpec,
    policy: str,
    seeds: range,
    workers: int,
    stress_factor: float,
    teacher: str = DEFAULT_TEACHER,
    make_env: str = DEFAULT_ENV,
    learner_overrides: dict | None = None,
    cache_teacher: bool = True,
) -> dict:
    """Teacher and learned skill on the same held-out seeds, nominal and stressed."""
    from skillforge.eval.parallel import EvalPool
    from skillforge.eval.regression import source_of
    from skillforge.eval.robustness import stress_spec, with_randomization

    from .dataset import load_object
    from .skill import learned_source

    env = load_object(make_env)
    srcs = {"teacher": source_of(load_object(teacher)), "learned": learned_source(policy, **(learner_overrides or {}))}
    conds = {"nominal": task, f"stress_x{stress_factor:g}": with_randomization(task, stress_spec(task.scene.randomization, stress_factor))}
    out: dict = {"task_id": task.id, "seeds": [seeds.start, seeds.stop], "policy": str(policy),
                 "learner_overrides": learner_overrides or {}, "results": {}}
    with EvalPool(workers) as pool:
        for cname, t in conds.items():
            for who, src in srcs.items():
                # The teacher's result is a pure function of (code, task, seeds): cache it.
                # The key covers the sim backend's source too, so a physics change invalidates it.
                key = hashlib.sha256((src.code + t.model_dump_json() + f"{seeds.start}:{seeds.stop}"
                                      + make_env + _sim_fingerprint()).encode()).hexdigest()[:16]
                cache = RUNS / "compare_cache" / f"{who}_{key}.json" if who == "teacher" and cache_teacher else None
                if cache is not None and cache.exists():
                    res = json.loads(cache.read_text())
                else:
                    t0 = time.perf_counter()
                    rep, eps = pool.evaluate(env, src, t, seeds)
                    res = {
                        "skill": f"{rep.skill}@{rep.skill_version}",
                        "n": rep.n_rollouts, "k": rep.n_success, "rate": rep.success_rate, "ci95": list(rep.ci95),
                        "failures": rep.failure_counts, "seconds": round(time.perf_counter() - t0, 1),
                        "per_seed": [bool(e.success) for e in eps],
                        "summary": rep.summary(),
                    }
                    if cache is not None:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps(res))
                out["results"].setdefault(cname, {})[who] = res
                print(f"  {cname:>12} {who:>8}: {res['summary']}", flush=True)
    for r in out["results"].values():
        te, le = r["teacher"], r["learned"]
        r["matches"] = le["rate"] >= te["ci95"][0]  # not below the teacher's Wilson lower bound
        r["beats"] = le["ci95"][0] > te["rate"]
        pairs = list(zip(te["per_seed"], le["per_seed"], strict=True))
        r["paired"] = {  # same seeds, so the disagreements are what the comparison rests on
            "both": sum(a and b for a, b in pairs),
            "teacher_only": sum(a and not b for a, b in pairs),
            "learned_only": sum(b and not a for a, b in pairs),
            "neither": sum(not a and not b for a, b in pairs),
        }
    shifted = [c for c in out["results"] if c != "nominal"]
    out["gate"] = {
        "rule": "learned success under shifted randomization >= teacher's Wilson 95% lower bound (matches); "
                "learned lower bound > teacher rate (beats)",
        "met": all(out["results"][c]["matches"] for c in shifted),
        "beats": all(out["results"][c]["beats"] for c in shifted),
    }
    return out


def cmd_eval(a: argparse.Namespace) -> dict:
    task = load_task(a.task)
    seeds = range(a.seed_start, a.seed_start + a.seeds)
    res = compare(task, a.policy, seeds, a.workers, a.stress, a.teacher, a.make_env,
                  {"ensemble_m": a.ensemble_m}, cache_teacher=not a.no_cache)
    out = Path(a.out) if a.out else Path(a.policy).parent / "compare.json"
    out.write_text(json.dumps(res, indent=2))
    for cname, r in res["results"].items():
        te, le, pr = r["teacher"], r["learned"], r["paired"]
        print(f"{cname:>12}: teacher {te['k']}/{te['n']} ({te['rate']:.0%}, CI {te['ci95'][0]:.0%}-{te['ci95'][1]:.0%})"
              f"  learned {le['k']}/{le['n']} ({le['rate']:.0%}, CI {le['ci95'][0]:.0%}-{le['ci95'][1]:.0%})"
              f"  matches={r['matches']} beats={r['beats']}"
              f"  paired: teacher-only {pr['teacher_only']}, learned-only {pr['learned_only']}")
    print(f"gate met: {res['gate']['met']} (beats: {res['gate']['beats']}); wrote {out}")
    return res


# ---------------------------------------------------------------- argparse


def _ensemble(v: str) -> float | None:
    return None if v.lower() in ("none", "off") else float(v)


def _sync_tol(v: str) -> float | str | None:
    return "auto" if v == "auto" else _ensemble(v)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m skillforge.learning", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="franka_pick_cube_to_bin")
    common.add_argument("--teacher", default=DEFAULT_TEACHER, type=resolve_spec,
                        help="'module:Class' or 'path/to/skill.py[:Class]' (default: %(default)s)")
    common.add_argument("--make-env", default=DEFAULT_ENV)
    common.add_argument("--workers", type=int, default=4)

    tr = argparse.ArgumentParser(add_help=False)
    tr.add_argument("--n-hist", type=int, default=2)
    tr.add_argument("--horizon", type=int, default=10)
    tr.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--train-seed", type=int, default=0)
    tr.add_argument("--threads", type=int, default=4)
    tr.add_argument("--max-minutes", type=float, default=None)

    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", parents=[common])
    c.add_argument("--name", required=True, help="dataset name under runs/datasets/, or a path")
    c.add_argument("--seeds", type=int, default=500)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--dart", type=float, default=0.0, help="DART burst noise std in metres (0 = off)")
    c.add_argument("--dart-p", type=float, default=0.3)
    c.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--collect-stress", type=float, default=None,
                   help="collect under T4's stress randomization x this factor (a different task sha: own dataset)")

    t = sub.add_parser("train", parents=[tr])
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)

    d = sub.add_parser("dagger", parents=[common, tr])
    d.add_argument("--data", required=True, help="dataset to aggregate into")
    d.add_argument("--policy", required=True, help="initial checkpoint")
    d.add_argument("--rounds", type=int, default=3)
    d.add_argument("--start-round", type=int, default=1)
    d.add_argument("--seeds", type=int, default=200, help="learner rollouts per round")
    d.add_argument("--beta", type=float, default=0.0)
    d.add_argument("--relabel", choices=("shadow", "phase_sync"), default="shadow",
                   help="shadow: unsynced shadow teacher (+ sync gate); phase_sync: expert.PhaseSyncExpert")
    d.add_argument("--sync-tol", type=_sync_tol, default="auto",
                   help="metres; labels after teacher/learner disagree longer than --sync-patience are dropped "
                        "('none' keeps all; 'auto': 0.03 for shadow, none for phase_sync)")
    d.add_argument("--extra-data", nargs="*", default=[], help="more datasets to train on (not appended to)")
    d.add_argument("--seed-offset", type=int, default=0, help="round r rolls out seeds 100000*r + offset ..")
    d.add_argument("--sync-patience", type=int, default=3)
    d.add_argument("--ensemble-m", type=_ensemble, default=0.1)
    d.add_argument("--out-dir", default=None)
    d.add_argument("--prefix", default="bc_dagger")
    d.add_argument("--collect-stress", type=float, default=None,
                   help="roll the learner out under stress randomization x this factor (dataset must match)")

    e = sub.add_parser("eval", parents=[common], aliases=["compare"])
    e.add_argument("--policy", required=True)
    e.add_argument("--seeds", type=int, default=200)
    e.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    e.add_argument("--stress", type=float, default=2.0)
    e.add_argument("--ensemble-m", type=_ensemble, default=0.1)
    e.add_argument("--out", default=None)
    e.add_argument("--no-cache", action="store_true", help="re-run the teacher instead of reusing runs/compare_cache/")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    {"collect": cmd_collect, "train": cmd_train, "dagger": cmd_dagger, "eval": cmd_eval, "compare": cmd_eval}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
