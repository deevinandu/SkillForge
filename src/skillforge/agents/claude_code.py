"""ClaudeCodeImprover: a Claude Code agent edits the skill in a scratch workspace.

No Anthropic API key is needed: it drives the `claude` CLI headless (`claude -p`), which uses
whatever auth that CLI already has (a Claude Code login or cloud session). Each `propose()`:

1. writes a workspace: `skill.py` (the current version), `REPORT.md` (the eval report and worst
   failures), `CONTEXT.md` (contracts, primitives, reference skill, task), `task.json`, and
   `./dev_eval`, a script that evaluates `skill.py` in the subprocess sandbox on DEV seeds;
2. runs Claude Code there with tools limited to reading and editing files and running
   `./dev_eval`, so it can iterate on its own change before handing it back;
3. reads `skill.py` and `RATIONALE.md` back, checks the file in the sandbox, and, if the check
   fails, resumes the same Claude Code session once with the error.

Dev seeds are disjoint from the loop's gate seeds (the loop scores on 0..rollouts-1), so the agent
cannot tune to the seeds it is judged on. The loop's keep-the-best gate still decides acceptance.
Spend comes from the CLI's own `total_cost_usd` and is capped by `--max-budget-usd` per call and by
`budget_usd` across calls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillforge.contracts import EvalReport, SimEnv, SkillSource, TaskSpec

from .budget import BudgetExceeded, CallUsage, UsageMeter
from .lessons import (
    LESSON_REPORT_FILE,
    LESSON_REPORT_INSTRUCTIONS,
    TOPICS,
    LessonReport,
    parse_lesson_report,
)
from .library import bump_version
from .llm_improver import (
    DEFAULT_MODEL,
    FAILURE_GLOSSARY,
    ImproverError,
    build_shared_context,
    find_entrypoint,
    format_failures,
)
from .sandbox import check_source, env_ref

DEV_SEED_BASE = 100_000
ALLOWED_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "Bash(./dev_eval)", "Bash(./dev_eval:*)")

DEV_EVAL = '''#!{python}
"""Evaluate ./skill.py on the DEV seeds in the SkillForge sandbox. Usage: ./dev_eval [n_seeds]"""
import json, sys
from pathlib import Path
from skillforge.agents.llm_improver import find_entrypoint
from skillforge.agents.sandbox import evaluate_source
from skillforge.contracts import SkillSource, TaskSpec

here = Path(__file__).resolve().parent
cfg = json.loads((here / "dev_eval.json").read_text())
task = TaskSpec.model_validate_json((here / "task.json").read_text())
code = (here / "skill.py").read_text()
try:
    entry = find_entrypoint(code, cfg["entrypoint"])
except (ValueError, SyntaxError) as e:
    sys.exit(f"skill.py does not parse: {{e}}")
src = SkillSource(name=cfg["name"], version="dev", code=code, entrypoint=entry)
n = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["n_seeds"]
n = max(1, min(n, cfg["max_seeds"]))
seeds = [cfg["seed_base"] + i for i in range(n)]
res = evaluate_source(task, src, cfg["make_env"], seeds, timeout_s=cfg["timeout_s"])
rep = res.report
print(rep.summary())
if res.stage != "eval" or not res.ok:
    print(f"sandbox stage={{res.stage}} error:\\n{{(res.error or '')[-3000:]}}")
for f in rep.failures[:10]:
    rnd = ", ".join(f"{{k}}={{v:.3g}}" for k, v in sorted(f.randomization.items()))
    print(f"- seed {{f.seed}} [{{f.tag}}] {{f.detail[:300]}} | {{rnd}}")
'''

INSTRUCTIONS = """\
You are improving a robot skill for SkillForge. Work only inside this directory.

Files:
- skill.py: the current skill, {name}@{version}, class `{entrypoint}`. Edit THIS file.
- REPORT.md: how it did on the gate seeds, with the worst failures and their randomization.
- CONTEXT.md: contracts, the primitive library, the reference skill and the task spec.
- ./dev_eval [n]: runs skill.py in the simulator sandbox on {n_dev} DEV seeds (up to {max_dev})
  and prints success rate, failure tags and failing seeds. Each run takes a while; use it.

Goal: raise the success rate under the task's randomization. Find the cause of the most common
failures in REPORT.md, change the skill to fix it, and check with ./dev_eval that the dev success
rate went up and nothing else broke. Keep the class name `{entrypoint}` and set its `version`
attribute to "{new_version}".

Rules for skill code: one self-contained file; it may import numpy, the standard library,
skillforge.contracts and skillforge.skills; no network, files, subprocesses or randomness; it must
not special-case seeds or randomization values (it only sees observations, so it cannot anyway).
{notes}{lessons}
When done, write RATIONALE.md: at most five sentences on what you changed, which failure it
targets, and the dev success rate before and after. Then stop.
"""


@dataclass
class CLIResult:
    ok: bool
    session_id: str | None
    cost_usd: float
    usage: dict[str, Any]
    text: str
    error: str = ""
    duration_s: float = 0.0
    num_turns: int = 0


Runner = Callable[[list[str], Path, float], subprocess.CompletedProcess]


def _default_runner(cmd: list[str], cwd: Path, timeout_s: float) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):  # a nested CLI must not think it is the parent
        env.pop(k, None)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)


def parse_cli_json(stdout: str) -> CLIResult:
    """The last JSON object on stdout from `claude -p --output-format json`."""
    doc = None
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if doc is None:
        try:
            doc = json.loads(stdout)
        except json.JSONDecodeError:
            return CLIResult(False, None, 0.0, {}, "", error=f"no JSON result on stdout: {stdout[-1500:]}")
    subtype = str(doc.get("subtype", "success"))
    is_error = bool(doc.get("is_error")) or subtype != "success"
    error = ""
    if is_error:
        # The CLI can report `is_error` with subtype "success" (e.g. an expired login, an API error);
        # the real message is then in `result`, the reason in `terminal_reason`.
        if subtype != "success":
            error = subtype
        else:
            why = str(doc.get("terminal_reason") or "error")
            msg = str(doc.get("result") or "").strip()
            error = f"{why}: {msg[:1500]}" if msg else why
    return CLIResult(
        ok=not is_error,
        session_id=doc.get("session_id"),
        cost_usd=float(doc.get("total_cost_usd") or 0.0),
        usage=doc.get("usage") or {},
        text=str(doc.get("result") or ""),
        error=error,
        duration_s=float(doc.get("duration_ms") or 0) / 1000,
        num_turns=int(doc.get("num_turns") or 0),
    )


class ClaudeCodeImprover:
    """SkillImprover backed by a headless Claude Code agent. See the module docstring."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        budget_usd: float | None = None,
        per_call_budget_usd: float = 5.0,
        make_env: Callable[[TaskSpec], SimEnv] | str = "skillforge.sim:make_env",
        dev_seeds: int = 30,
        max_dev_seeds: int = 100,
        dev_timeout_s: float = 900.0,
        max_turns: int = 60,
        call_timeout_s: float = 3600.0,
        check_timeout_s: float = 60.0,
        max_fix_attempts: int = 1,
        max_failure_cases: int = 10,
        workdir: str | Path | None = None,
        claude_bin: str = "claude",
        runner: Runner | None = None,
        python: str = sys.executable,
    ):
        self.model = model
        self.usage = UsageMeter(budget_usd=budget_usd)
        self.per_call_budget_usd = per_call_budget_usd
        self.make_env = env_ref(make_env)
        self.dev_seeds = dev_seeds
        self.max_dev_seeds = max_dev_seeds
        self.dev_timeout_s = dev_timeout_s
        self.max_turns = max_turns
        self.call_timeout_s = call_timeout_s
        self.check_timeout_s = check_timeout_s
        self.max_fix_attempts = max_fix_attempts
        self.max_failure_cases = max_failure_cases
        self.workdir = Path(workdir) if workdir else None
        self.python = python
        if runner is None:
            if shutil.which(claude_bin) is None:
                raise ImproverError(f"`{claude_bin}` CLI not found on PATH; install Claude Code or use --improver fake")
            runner = _default_runner
        self.claude_bin = claude_bin
        self.runner = runner
        self.notes: list[str] = []
        # scoped lessons (docs/LESSONS.md): set by run_loop / curate before each propose()
        self.lessons_text: str = ""  # promoted lessons for this robot, rendered; written to LESSONS.md
        self.directive: str = ""  # an extra instruction (the verifier's "reproduce", the checker's "apply only")
        self.lesson_report: LessonReport | None = None  # LESSONS.json from the last propose()
        self.calls: list[dict] = []  # per CLI call: workspace, cost, turns, duration, session

    # -- workspace -------------------------------------------------------------

    def _workspace(self, task: TaskSpec, current: SkillSource) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        if self.workdir:
            base = self.workdir / task.id / f"{stamp}-{current.version}"
            ws, n = base, 1
            while ws.exists():  # a rejected round re-proposes from the same version, maybe within a second
                ws, n = base.with_name(f"{base.name}-{n}"), n + 1
            ws.mkdir(parents=True, exist_ok=False)
        else:
            ws = Path(tempfile.mkdtemp(prefix=f"sf-{task.id}-{current.version}-"))
        return ws

    def _write_workspace(self, ws: Path, task: TaskSpec, current: SkillSource, report: EvalReport, new_version: str) -> str:
        (ws / "skill.py").write_text(current.code)
        (ws / "task.json").write_text(task.model_dump_json(indent=1))
        (ws / "CONTEXT.md").write_text(build_shared_context(task))
        glossary = "\n".join(f"- {k}: {v}" for k, v in FAILURE_GLOSSARY.items())
        (ws / "REPORT.md").write_text(
            f"# Gate eval of {current.name}@{current.version}\n\n{report.summary()}\n\n"
            f"## Worst failures\n{format_failures(report, self.max_failure_cases)}\n\n## Failure tags\n{glossary}\n"
        )
        cfg = {
            "name": current.name,
            "entrypoint": current.entrypoint,
            "make_env": self.make_env,
            "seed_base": DEV_SEED_BASE,
            "n_seeds": self.dev_seeds,
            "max_seeds": self.max_dev_seeds,
            "timeout_s": self.dev_timeout_s,
        }
        (ws / "dev_eval.json").write_text(json.dumps(cfg, indent=1))
        script = ws / "dev_eval"
        script.write_text(DEV_EVAL.format(python=self.python))
        script.chmod(0o755)
        lessons = ""
        if self.lessons_text:
            (ws / "LESSONS.md").write_text(self.lessons_text)
            lessons += "\nLESSONS.md holds lessons from earlier repairs (other tasks and robots); read it before changing anything.\n"
        if self.directive:
            lessons += f"\n{self.directive}\n"
        lessons += "\n" + LESSON_REPORT_INSTRUCTIONS.format(topics="|".join(TOPICS), robot=task.scene.robot.name)
        notes = ""
        if self.notes:
            notes = "\nEarlier attempts in this run (learn from them):\n" + "\n".join(f"- {n}" for n in self.notes[-10:]) + "\n"
        return INSTRUCTIONS.format(
            name=current.name,
            version=current.version,
            entrypoint=current.entrypoint,
            n_dev=self.dev_seeds,
            max_dev=self.max_dev_seeds,
            new_version=new_version,
            notes=notes,
            lessons=lessons,
        )

    # -- CLI -------------------------------------------------------------------

    def _call(self, ws: Path, prompt: str, resume: str | None = None) -> CLIResult:
        remaining = None if self.usage.budget_usd is None else self.usage.budget_usd - self.usage.usd
        cap = self.per_call_budget_usd if remaining is None else min(self.per_call_budget_usd, remaining)
        if cap <= 0.01:
            raise BudgetExceeded(f"claude-code call refused: spent ${self.usage.usd:.2f} of ${self.usage.budget_usd:.2f}")
        cmd = [
            self.claude_bin, "-p", prompt,
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--max-budget-usd", f"{cap:.2f}",
            "--permission-mode", "dontAsk",
            "--allowedTools", *ALLOWED_TOOLS,
            "--strict-mcp-config",
        ]  # fmt: skip
        if resume:
            cmd += ["--resume", resume]
        t0 = time.monotonic()
        try:
            proc = self.runner(cmd, ws, self.call_timeout_s)
        except subprocess.TimeoutExpired:
            raise ImproverError(f"claude-code timed out after {self.call_timeout_s:.0f}s in {ws}") from None
        res = parse_cli_json(proc.stdout or "")
        if not res.error and proc.returncode not in (0, None) and not res.ok:
            res.error = f"exit {proc.returncode}: {(proc.stderr or '')[-1500:]}"
        u = res.usage
        call = CallUsage(
            role="claude-code",
            model=self.model,
            input_tokens=int(u.get("input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
            usd=res.cost_usd,  # the CLI's own figure, not a list-price estimate
        )
        self.usage.calls.append(call)
        self.calls.append(
            {"workspace": str(ws), "session_id": res.session_id, "usd": res.cost_usd, "turns": res.num_turns,
             "duration_s": round(time.monotonic() - t0, 1), "ok": res.ok, "error": res.error}
        )  # fmt: skip
        return res

    # -- SkillImprover ---------------------------------------------------------

    def record_outcome(self, version: str, accepted: bool, reason: str) -> None:
        """Called by `run_loop` after the gate: the verdict goes into the next round's notes."""
        self.notes.append(f"{version} gate verdict: {'ACCEPTED' if accepted else 'REJECTED'} ({reason[:200]})")

    def propose(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> SkillSource:
        new_version = bump_version(current.version)
        ws = self._workspace(task, current)
        prompt = self._write_workspace(ws, task, current, report, new_version)
        self.lesson_report = None
        res = self._call(ws, prompt)
        session = res.session_id
        last_err = res.error
        for attempt in range(1 + self.max_fix_attempts):
            code = (ws / "skill.py").read_text()
            if code == current.code:
                last_err = last_err or "the agent did not change skill.py"
            else:
                try:
                    entry = find_entrypoint(code, current.entrypoint)
                    rat = ws / "RATIONALE.md"
                    cand = SkillSource(
                        name=current.name,
                        version=new_version,
                        code=code,
                        entrypoint=entry,
                        rationale=(rat.read_text().strip() if rat.exists() else res.text.strip())[:2000],
                        parent_version=current.version,
                    )
                    chk = check_source(task, cand, self.make_env, timeout_s=self.check_timeout_s)
                    if chk.ok:
                        self.lesson_report = _read_lesson_report(ws)
                        self.notes.append(f"{new_version} (from {current.version} at {report.success_rate:.0%}): {cand.rationale[:300]}")
                        return cand
                    last_err = f"sandbox {chk.stage} failed:\n{chk.error}"
                except (ValueError, SyntaxError) as e:
                    last_err = f"{type(e).__name__}: {e}"
            if attempt >= self.max_fix_attempts or not session:
                break
            fix = (
                f"skill.py did not pass validation:\n```\n{last_err[-3000:]}\n```\n"
                "Fix skill.py (run ./dev_eval to confirm it loads and runs), update RATIONALE.md, then stop."
            )
            res = self._call(ws, fix, resume=session)
            session = res.session_id or session
        self.notes.append(f"attempt from {current.version} produced no valid skill: {last_err[:200]}")
        raise ImproverError(f"claude-code produced no valid skill (workspace {ws}); last error: {last_err[-1500:]}")


def dev_seed_list(n: int) -> Sequence[int]:
    return range(DEV_SEED_BASE, DEV_SEED_BASE + n)


def _read_lesson_report(ws: Path) -> LessonReport | None:
    """LESSONS.json from the workspace, or None if missing or malformed (a lesson is optional)."""
    f = ws / LESSON_REPORT_FILE
    if not f.exists():
        return None
    try:
        return parse_lesson_report(f.read_text())
    except Exception:
        return None
