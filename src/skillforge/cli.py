"""`skillforge` command line. Currently one subcommand:

    skillforge forge tasks/X.json --seed-skill pick_place --rounds 8 --rollouts 100 \\
        --improver claude|fake|null [--budget-usd 5]

Prints one line per round and writes runs/forge/<timestamp>/history.json (schema in
skillforge.agents.loop) plus every evaluated version under runs/library/<task_id>/.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from skillforge.contracts import SkillSource, TaskSpec


def load_seed_skill(ref: str) -> SkillSource:
    """A skill module name in skillforge.skills (e.g. 'pick_place'), or a path to a .py file.

    A path may carry the entrypoint as 'path.py:ClassName'.
    """
    path, _, entry = ref.partition(":")
    p = Path(path)
    if p.suffix == ".py" and p.exists():
        code = p.read_text()
        name = p.stem
    else:
        mod = importlib.import_module(f"skillforge.skills.{path}")
        code = inspect.getsource(mod)
        name = path
    from skillforge.agents.llm_improver import find_entrypoint

    camel = "".join(w.capitalize() for w in name.split("_"))
    entry = entry or find_entrypoint(code, camel)
    version = "0.1.0"
    if p.suffix != ".py":
        cls = getattr(mod, entry, None)
        version = str(getattr(cls, "version", version))
        name = str(getattr(cls, "name", name))
    return SkillSource(name=name, version=version, code=code, entrypoint=entry, rationale="seed skill")


def make_improver(kind: str, args: argparse.Namespace):
    from skillforge.agents import ClaudeImprover, FakeImprover, NullImprover

    if kind == "null":
        return NullImprover()
    if kind == "fake":
        return FakeImprover()
    return ClaudeImprover(model=args.model, budget_usd=args.budget_usd, make_env=args.env)


def _fmt_round(rec) -> str:
    rep = rec.report
    lo, hi = rep.ci95
    top = ", ".join(f"{k}={v}" for k, v in sorted(rep.failure_counts.items(), key=lambda kv: -kv[1])[:3]) or "none"
    mark = "ACCEPT" if rec.accepted else "reject"
    u = rec.usage
    tok = f" tok in/out {int(u['input_tokens'] + u['cache_read_input_tokens'] + u['cache_creation_input_tokens'])}/{int(u['output_tokens'])}" if u.get("calls") else ""
    return (
        f"round {rec.round:>2}  {rec.source.version:<12} {rep.n_success:>4}/{rep.n_rollouts:<4} "
        f"{rep.success_rate:>4.0%} [{lo:.0%}-{hi:.0%}]  {mark:<6} {rec.reason[:60]:<60} "
        f"fails: {top}{tok}  ${rec.usd_total:.4f}"
    )


def cmd_forge(args: argparse.Namespace) -> int:
    from skillforge.agents import MissingCredentials, SkillLibrary, run_loop

    task = TaskSpec.model_validate(json.loads(Path(args.task).read_text()))
    seed = load_seed_skill(args.seed_skill)
    try:
        improver = make_improver(args.improver, args)
    except MissingCredentials as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / stamp
    n = 1
    while out.exists():
        out, n = Path(args.out) / f"{stamp}-{n}", n + 1
    meta = {
        "task_file": str(args.task),
        "seed_skill": args.seed_skill,
        "improver": args.improver,
        "model": args.model if args.improver == "claude" else None,
        "rounds": args.rounds,
        "rollouts": args.rollouts,
        "target": args.target,
        "budget_usd": args.budget_usd,
        "timeout_s": args.timeout,
        "env": args.env,
    }
    print(f"forge {task.id}: seed {seed.name}@{seed.version}, improver {args.improver}, -> {out}/history.json")
    result = run_loop(
        task,
        seed,
        improver,
        args.env,
        rounds=args.rounds,
        rollouts=args.rollouts,
        target=args.target,
        timeout_s=args.timeout,
        library=SkillLibrary(args.library),
        history_path=out / "history.json",
        on_round=lambda rec: print(_fmt_round(rec), flush=True),
        meta=meta,
    )
    b = result.best
    print(f"stopped: {result.stopped}; best {b.source.version}: {b.report.summary()}")
    usage = getattr(improver, "usage", None)
    if usage is not None and usage.calls:
        t = usage.totals()
        print(
            f"usage: {int(t['calls'])} calls, in {int(t['input_tokens'])} (+cache write {int(t['cache_creation_input_tokens'])}, "
            f"cache read {int(t['cache_read_input_tokens'])}), out {int(t['output_tokens'])}, est ${t['usd']:.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skillforge", description="CI/CD for robot behaviors")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forge", help="evaluate a seed skill and let an improver rewrite it")
    f.add_argument("task", help="path to a TaskSpec JSON file")
    f.add_argument("--seed-skill", default="pick_place", help="skillforge.skills module name, or path.py[:Class]")
    f.add_argument("--rounds", type=int, default=8, help="evaluations, seed included")
    f.add_argument("--rollouts", type=int, default=100, help="seeds per evaluation (the same seeds every round)")
    f.add_argument("--target", type=float, default=0.9, help="stop once an accepted version reaches this success rate")
    f.add_argument("--improver", choices=("claude", "fake", "null"), default="claude")
    f.add_argument("--model", default="claude-opus-5-5")
    f.add_argument("--budget-usd", type=float, default=None, help="hard cap on estimated API spend")
    f.add_argument("--timeout", type=float, default=600.0, help="sandbox wall-clock limit per evaluation, seconds")
    f.add_argument("--env", default="skillforge.sim:make_env", help="env factory as module:attr")
    f.add_argument("--library", default="runs/library")
    f.add_argument("--out", default="runs/forge")
    f.set_defaults(func=cmd_forge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
