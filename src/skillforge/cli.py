"""`skillforge` command line. Currently one subcommand:

    skillforge forge tasks/X.json --seed-skill pick_place --rounds 8 --rollouts 100 \\
        --improver claude|claude-code|fake|null [--budget-usd 5] [--gate paired|ci --gate-alpha 0.1]

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
    if kind == "claude-code":
        from skillforge.agents.claude_code import ClaudeCodeImprover

        return ClaudeCodeImprover(
            model=args.model,
            budget_usd=args.budget_usd,
            per_call_budget_usd=args.per_call_budget_usd,
            make_env=args.env,
            dev_seeds=args.dev_seeds,
            workdir=Path(args.out).parent / "claude_code",
        )
    return ClaudeImprover(model=args.model, budget_usd=args.budget_usd, make_env=args.env)


def _fmt_round(rec) -> str:
    rep = rec.report
    lo, hi = rep.ci95
    top = ", ".join(f"{k}={v}" for k, v in sorted(rep.failure_counts.items(), key=lambda kv: -kv[1])[:3]) or "none"
    mark = "ACCEPT" if rec.accepted else "reject"
    u = rec.usage
    pc = rec.paired
    pair = f"  paired vs {pc['vs_version']} +{pc['cand_only']}/-{pc['best_only']} p={pc['p_value']:.3g}" if pc else ""
    tok = f" tok in/out {int(u['input_tokens'] + u['cache_read_input_tokens'] + u['cache_creation_input_tokens'])}/{int(u['output_tokens'])}" if u.get("calls") else ""
    return (
        f"round {rec.round:>2}  {rec.source.version:<12} {rep.n_success:>4}/{rep.n_rollouts:<4} "
        f"{rep.success_rate:>4.0%} [{lo:.0%}-{hi:.0%}]  {mark:<6} {rec.reason[:60]:<60} "
        f"fails: {top}{pair}{tok}  ${rec.usd_total:.4f}"
    )


def cmd_forge(args: argparse.Namespace) -> int:
    from skillforge.agents import MissingCredentials, SkillLibrary, run_loop
    from skillforge.agents.lessons import LessonStore

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
        "model": args.model if args.improver in ("claude", "claude-code") else None,
        "rounds": args.rounds,
        "rollouts": args.rollouts,
        "target": args.target,
        "budget_usd": args.budget_usd,
        "timeout_s": args.timeout,
        "env": args.env,
        "gate": args.gate,
        "gate_alpha": args.gate_alpha,
        "per_call_budget_usd": args.per_call_budget_usd if args.improver == "claude-code" else None,
        "lessons": args.lessons,
        "lesson_policy": args.lesson_policy if args.lessons else None,
        "robot": args.robot or task.scene.robot.name,
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
        gate=args.gate,
        alpha=args.gate_alpha,
        lessons=LessonStore(args.lessons) if args.lessons else None,
        robot=args.robot,
        lesson_policy=args.lesson_policy,
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


def _seed_range(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def cmd_lessons(args: argparse.Namespace) -> int:
    from skillforge.agents import SkillLibrary
    from skillforge.agents.curate import Target, admit, check_scope, verify_lesson
    from skillforge.agents.lessons import LessonStore
    from skillforge.agents.scope import label_confusion

    store = LessonStore(args.dir)
    if args.action == "list":
        for les in store.all():
            scope = les.scope.label() if les.scope else "-"
            agent = les.agent_scope.label() if les.agent_scope else "-"
            useful = sum(g.verdict == "useful" for g in les.grades)
            print(f"{les.id:40s} {les.status:9s} scope {scope:22s} agent {agent:22s} "
                  f"struct {les.structural_scope or '-':14s} checks {len(les.scope_checks)} "
                  f"grades +{useful}/-{len(les.bad_grades())}  {les.source.robot}/{les.source.task_id}")
        conf = label_confusion(store.all())
        if conf:
            print("scope-label confusion (agent vs checking):", conf)
        return 0
    if args.action == "show":
        print(store.get(args.id).model_dump_json(indent=2))
        return 0
    if args.action == "admit":
        les = admit(store, args.id, args.method)
        print(f"{les.id}: {les.status}" + (f", scope {les.scope.label()}" if les.scope else ""))
        return 0 if les.status == "promoted" else 1

    def factory():
        return make_improver(args.improver, args)

    seeds = _seed_range(args.seeds)
    les = store.get(args.id)
    if args.action == "verify":
        task = TaskSpec.model_validate(json.loads(Path(args.task).read_text()))
        if task.id != les.source.task_id:
            print(f"error: lesson came from {les.source.task_id}, task file is {task.id}", file=sys.stderr)
            return 2
        parent, _ = SkillLibrary(args.library).load(task.id, les.source.parent_version)
        les = verify_lesson(store, les.id, task=task, parent=parent, make_env=args.env, improver_factory=factory,
                            seeds=seeds, attempts=args.attempts, alpha=args.gate_alpha, timeout_s=args.timeout)
        print(f"{les.id}: {les.status}")
        return 0
    targets = []
    for t in args.target:
        robot, task_file, skill_ref = t.split("=", 2)
        targets.append(Target(robot, TaskSpec.model_validate(json.loads(Path(task_file).read_text())), load_seed_skill(skill_ref)))
    les = check_scope(store, les.id, targets, make_env=args.env, improver_factory=factory, seeds=seeds,
                      alpha=args.gate_alpha, timeout_s=args.timeout)
    for c in les.scope_checks:
        print(f"{c.robot}/{c.task_id}: {c.verdict} (+{c.cand_only}/-{c.best_only}, p_gain {c.p_gain:.3g}, p_loss {c.p_loss:.3g})")
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
    f.add_argument("--improver", choices=("claude", "claude-code", "fake", "null"), default="claude",
                   help="claude: Anthropic API (needs ANTHROPIC_API_KEY); claude-code: headless Claude Code CLI, no API key")
    f.add_argument("--model", default="claude-opus-5-5")
    f.add_argument("--budget-usd", type=float, default=None, help="hard cap on estimated API spend")
    f.add_argument("--per-call-budget-usd", type=float, default=5.0, help="claude-code: cap per agent session")
    f.add_argument("--dev-seeds", type=int, default=30, help="claude-code: dev seeds the agent may evaluate on (disjoint from gate seeds)")
    f.add_argument("--gate", choices=("paired", "ci"), default="paired",
                   help="paired: more gate seeds solved than the best AND one-sided exact McNemar p < --gate-alpha; "
                        "ci: legacy, accept unless below the best's 95%% CI lower bound")
    f.add_argument("--gate-alpha", type=float, default=0.1, help="significance level of the paired gate")
    f.add_argument("--timeout", type=float, default=600.0, help="sandbox wall-clock limit per evaluation, seconds")
    f.add_argument("--env", default="skillforge.sim:make_env", help="env factory as module:attr")
    f.add_argument("--library", default="runs/library")
    f.add_argument("--out", default="runs/forge")
    f.add_argument("--lessons", default=None, help="scoped lesson library dir (docs/LESSONS.md); off when unset")
    f.add_argument("--lesson-policy", choices=("scoped", "share_all", "none"), default="scoped",
                   help="which promoted lessons the improver sees: general + this robot's, all, or none")
    f.add_argument("--robot", default=None, help="robot name for lesson scope (default: the task's robot.name)")
    f.set_defaults(func=cmd_forge)

    lp = sub.add_parser("lessons", help="curate the scoped lesson library: list, show, verify, check, admit")
    lp.add_argument("--dir", default="runs/lessons", help="lesson library dir")
    ls = lp.add_subparsers(dest="action", required=True)
    ls.add_parser("list", help="every lesson with status, scope and grades")
    sh = ls.add_parser("show", help="one lesson as JSON")
    sh.add_argument("id")
    for name, hlp in (("verify", "blind re-derivation by a fresh improver from the parent version"),
                      ("check", "apply the lesson to other robots' skills, paired score on each")):
        a = ls.add_parser(name, help=hlp)
        a.add_argument("id")
        a.add_argument("--improver", choices=("claude", "claude-code", "fake", "null"), default="claude-code")
        a.add_argument("--model", default="claude-opus-5-5")
        a.add_argument("--budget-usd", type=float, default=None)
        a.add_argument("--per-call-budget-usd", type=float, default=5.0)
        a.add_argument("--dev-seeds", type=int, default=30)
        a.add_argument("--seeds", default="1000-1049", help="seed range a-b, never gate or held-out seeds")
        a.add_argument("--env", default="skillforge.sim:make_env")
        a.add_argument("--timeout", type=float, default=600.0)
        a.add_argument("--gate-alpha", type=float, default=0.1)
        a.add_argument("--out", default="runs/lessons_work/forge", help="claude-code workspaces go next to this")
        if name == "verify":
            a.add_argument("--task", required=True, help="the lesson's source TaskSpec JSON")
            a.add_argument("--library", default="runs/library", help="skill library holding the parent version")
            a.add_argument("--attempts", type=int, default=2)
        else:
            a.add_argument("--target", action="append", required=True, metavar="ROBOT=TASK.json=SKILL",
                           help="another robot's task file and current skill (module or path.py[:Class]); repeatable")
    ad = ls.add_parser("admit", help="decide the scope of a verified lesson and promote it")
    ad.add_argument("id")
    ad.add_argument("--method", choices=("checking", "structure", "agent"), default="checking")
    lp.set_defaults(func=cmd_lessons)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
