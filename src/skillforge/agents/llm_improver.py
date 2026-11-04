"""ClaudeImprover: planner, coder and critic roles over the Anthropic Messages API.

One shared system prompt (role-agnostic preamble + frozen contract source + primitive API +
the TaskSpec) carries a cache breakpoint, so all three roles and every round reuse one cached
prefix; the per-round data goes in the user turn. Calls stream (long outputs) and use adaptive
thinking with an explicit effort. Server-side refusal fallbacks are on by default.

Needs `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_PROFILE`) and the `agents`
extra. Without credentials the constructor raises `MissingCredentials` before any call.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
from collections.abc import Callable
from typing import Any

from skillforge.contracts import EvalReport, SimEnv, SkillSource, TaskSpec

from .budget import UsageMeter, worst_case_usd
from .library import bump_version
from .sandbox import check_source

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_CRED_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")


class MissingCredentials(RuntimeError):
    pass


class ImproverError(RuntimeError):
    pass


FAILURE_GLOSSARY = {
    "timeout": "ran out of steps without success and never grasped anything",
    "grasp_miss": "closed the gripper with no object within grasp radius, and never grasped",
    "grasp_slip": "lost the object while carrying it",
    "collision": "a collision event was reported",
    "object_dropped": "the episode terminated (e.g. object fell off the table)",
    "wrong_placement": "grasped the object but timed out without it in the goal region",
    "ik_failure": "the controller could not reach the commanded pose",
    "exception": "the skill raised, failed to load, or the sandbox timed out/crashed",
    "terminated": "the env ended the episode early",
    "other": "unclassified",
}

PREAMBLE = """\
You are part of SkillForge, which writes closed-loop robot skills as Python code and hardens them
under domain randomization. You play one of three roles per request (planner, coder, critic); the
user turn says which. Ground truth is the evaluation report: success is judged on true simulator
state over a fixed set of seeds, so every change in success rate is the skill, not luck.

Rules for skill code:
- One self-contained Python file. It may import numpy, the standard library, `skillforge.contracts`
  and `skillforge.skills` (primitives). No network, no file or subprocess access, no randomness:
  a skill must be deterministic given (task, observations).
- It defines one class implementing the Skill protocol: class attributes `name` and `version`,
  `reset(self, task: TaskSpec) -> None` and `act(self, obs: Observation) -> Action`.
- Units are metres, kilograms, seconds, radians; quaternions are (w, x, y, z).
- Observations of object positions can be noisy and actions can be delayed; design for that.
"""


def _module_source(name: str) -> str | None:
    try:
        return inspect.getsource(importlib.import_module(name))
    except (ImportError, OSError, TypeError):
        return None


def build_shared_context(task: TaskSpec) -> str:
    """Deterministic: same task -> byte-identical text, so the cache prefix stays valid."""
    parts = [PREAMBLE, "## Failure tags\n" + "\n".join(f"- {k}: {v}" for k, v in FAILURE_GLOSSARY.items())]
    for mod in ("skillforge.contracts.specs", "skillforge.contracts.runtime", "skillforge.contracts.agents"):
        parts.append(f"## Contract source: {mod}\n```python\n{_module_source(mod)}\n```")
    prim = _module_source("skillforge.skills.primitives")
    parts.append(
        f"## Primitive library: skillforge.skills.primitives\n```python\n{prim}\n```"
        if prim
        else "## Primitive library\n(none published yet; write the motion logic yourself)"
    )
    example = _module_source("skillforge.skills.pick_place")
    if example:
        parts.append(f"## Reference skill: skillforge.skills.pick_place\n```python\n{example}\n```")
    parts.append("## Task (TaskSpec JSON)\n```json\n" + json.dumps(json.loads(task.model_dump_json()), indent=1, sort_keys=True) + "\n```")
    return "\n\n".join(parts)


def format_failures(report: EvalReport, n: int) -> str:
    rows = []
    for f in report.failures[:n]:
        rnd = ", ".join(f"{k}={v:.4g}" for k, v in sorted(f.randomization.items()))
        rows.append(f"- seed {f.seed} [{f.tag}] randomization: {{{rnd}}}; detail: {f.detail[:600]}")
    return "\n".join(rows) or "(no failures)"


_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)


def parse_code_reply(text: str) -> tuple[str, str]:
    """(code, rationale) from a reply with a fenced python block. Uses the LAST python block."""
    blocks = _FENCE.findall(text)
    if not blocks:
        raise ValueError("reply has no ```python fenced block")
    code = blocks[-1].strip() + "\n"
    m = re.search(r"RATIONALE:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    rationale = m.group(1) if m else _FENCE.sub("", text)
    rationale = _FENCE.sub("", rationale).strip()
    return code, rationale[:2000]


def find_entrypoint(code: str, preferred: str) -> str:
    """Static guess only; the sandbox's protocol check is what actually validates the class.

    Order: the preferred name; else the first class defining reset() and act(); else the last
    top-level class (a subclass that inherits both).
    """
    classes = [n for n in ast.parse(code).body if isinstance(n, ast.ClassDef)]
    if not classes:
        raise ValueError("no top-level class in the file")
    if preferred in {c.name for c in classes}:
        return preferred
    full = [c.name for c in classes if {"act", "reset"} <= {f.name for f in c.body if isinstance(f, ast.FunctionDef)}]
    return full[0] if full else classes[-1].name


def parse_verdict(text: str) -> bool | None:
    m = re.findall(r"VERDICT:\s*(ACCEPT|REVERT)", text, re.IGNORECASE)
    return None if not m else m[-1].upper() == "ACCEPT"


def _text(content: list[Any]) -> str:
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", "") == "text")


class ClaudeImprover:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        budget_usd: float | None = None,
        make_env: Callable[[TaskSpec], SimEnv] | str = "skillforge.sim:make_env",
        max_fix_attempts: int = 2,
        check_timeout_s: float = 60.0,
        coder_max_tokens: int = 32000,
        aux_max_tokens: int = 8000,
        coder_effort: str = "high",
        aux_effort: str = "medium",
        use_fallbacks: bool = True,
        max_failure_cases: int = 8,
    ):
        if client is None:
            if not any(os.environ.get(v) for v in _CRED_VARS):
                raise MissingCredentials(
                    "ClaudeImprover needs Anthropic credentials: set ANTHROPIC_API_KEY (or "
                    "ANTHROPIC_AUTH_TOKEN / ANTHROPIC_PROFILE). For offline runs use --improver fake."
                )
            try:
                import anthropic
            except ImportError as e:
                raise MissingCredentials('the Anthropic SDK is not installed: pip install -e ".[agents]"') from e
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.usage = UsageMeter(budget_usd=budget_usd)
        self.make_env = make_env
        self.max_fix_attempts = max_fix_attempts
        self.check_timeout_s = check_timeout_s
        self.coder_max_tokens = coder_max_tokens
        self.aux_max_tokens = aux_max_tokens
        self.coder_effort = coder_effort
        self.aux_effort = aux_effort
        self.use_fallbacks = use_fallbacks
        self.max_failure_cases = max_failure_cases
        self._ctx: dict[str, str] = {}
        self.notes: list[str] = []  # critic verdicts and past attempts, fed to the next planner
        self.transcript: list[dict] = []  # role, prompt, reply text: for debugging, not persisted

    # -- API ------------------------------------------------------------------

    def _system(self, task: TaskSpec) -> list[dict]:
        if task.id not in self._ctx:
            self._ctx[task.id] = build_shared_context(task)
        return [{"type": "text", "text": self._ctx[task.id], "cache_control": {"type": "ephemeral"}}]

    def _call(self, role: str, task: TaskSpec, messages: list[dict], max_tokens: int, effort: str) -> list[Any]:
        system = self._system(task)
        chars = len(system[0]["text"]) + sum(len(str(m["content"])) for m in messages)
        self.usage.check(worst_case_usd(self.model, chars, max_tokens), role)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        )
        if self.use_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        with self.client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        self.usage.record(role, self.model, getattr(msg, "model", None), msg.usage)
        self.transcript.append({"role": role, "prompt": messages[-1]["content"], "reply": _text(msg.content)})
        if msg.stop_reason == "refusal":
            raise ImproverError(f"{role} request was declined (refusal): {getattr(msg, 'stop_details', None)}")
        if msg.stop_reason == "max_tokens":
            raise ImproverError(f"{role} reply truncated at max_tokens={max_tokens}")
        return list(msg.content)

    # -- roles ----------------------------------------------------------------

    def plan(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> str:
        history = "\n".join(f"- {n}" for n in self.notes[-6:]) or "(first round)"
        prompt = f"""ROLE: planner.
Decide which failure to fix first in the skill below, and why. Prefer the failure mode that costs
the most rollouts and that the code can plausibly fix; use the randomization values to find the
condition that triggers it. Do not write code. Reply in under 250 words, ending with a numbered
list of concrete code changes.

## Current skill {current.name}@{current.version} (entrypoint {current.entrypoint})
```python
{current.code}
```

## Eval report
{report.summary()}

## Worst failures
{format_failures(report, self.max_failure_cases)}

## Earlier rounds
{history}
"""
        return _text(self._call("planner", task, [{"role": "user", "content": prompt}], self.aux_max_tokens, self.aux_effort))

    def code(self, task: TaskSpec, current: SkillSource, report: EvalReport, plan: str) -> SkillSource:
        prompt = f"""ROLE: coder.
Rewrite the skill to carry out the plan. Keep what already works; change what the plan targets.
Keep the class name `{current.entrypoint}` and set its `version` attribute to "{bump_version(current.version)}".

Reply with exactly one complete file in a ```python fenced block, then a line `RATIONALE:` followed
by at most five sentences on what changed and which failure it targets.

## Plan
{plan}

## Current skill {current.name}@{current.version}
```python
{current.code}
```

## Eval report
{report.summary()}

## Worst failures
{format_failures(report, self.max_failure_cases)}
"""
        messages: list[dict] = [{"role": "user", "content": prompt}]
        last_err = ""
        for attempt in range(1 + self.max_fix_attempts):
            content = self._call("coder", task, messages, self.coder_max_tokens, self.coder_effort)
            reply = _text(content)
            try:
                code, rationale = parse_code_reply(reply)
                entry = find_entrypoint(code, current.entrypoint)
                cand = SkillSource(
                    name=current.name,
                    version=bump_version(current.version),
                    code=code,
                    entrypoint=entry,
                    rationale=rationale,
                    parent_version=current.version,
                )
                res = check_source(task, cand, self.make_env, timeout_s=self.check_timeout_s)
                if res.ok:
                    return cand
                last_err = f"sandbox {res.stage} failed:\n{res.error}"
            except (ValueError, SyntaxError) as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt < self.max_fix_attempts:
                messages += [
                    {"role": "assistant", "content": content},  # append-only history
                    {
                        "role": "user",
                        "content": f"That file did not pass validation (attempt {attempt + 1}):\n```\n{last_err[-3000:]}\n```\n"
                        "Fix it and reply again with one complete ```python block and a RATIONALE: line.",
                    },
                ]
        raise ImproverError(f"coder produced no valid skill after {1 + self.max_fix_attempts} attempts; last error: {last_err[-1500:]}")

    def propose(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> SkillSource:
        plan = self.plan(task, current, report)
        cand = self.code(task, current, report, plan)
        self.notes.append(f"{cand.version} (from {current.version} at {report.success_rate:.0%}): {cand.rationale[:300]}")
        return cand

    def review(
        self, task: TaskSpec, parent: SkillSource, parent_report: EvalReport, cand: SkillSource, cand_report: EvalReport
    ) -> tuple[bool, str]:
        """Critic. Called only after the statistical gate passed; it can revert, never force accept."""
        prompt = f"""ROLE: critic.
A new skill version passed the statistical gate. Compare the reports and the code change and decide
whether to keep it. Revert if the gain looks like an artifact (e.g. it special-cases seeds or
randomization values, reads anything other than observations, or trades a rare failure for a
worse one). Reply in under 150 words and end with exactly `VERDICT: ACCEPT` or `VERDICT: REVERT`.

## Parent {parent.version}: {parent_report.summary()}
## Candidate {cand.version}: {cand_report.summary()}
## Candidate rationale
{cand.rationale}

## Candidate failures
{format_failures(cand_report, self.max_failure_cases)}

## Candidate code
```python
{cand.code}
```
"""
        text = _text(self._call("critic", task, [{"role": "user", "content": prompt}], self.aux_max_tokens, self.aux_effort))
        verdict = parse_verdict(text)
        ok = True if verdict is None else verdict
        why = text.strip()[-1500:] + ("" if verdict is not None else "\n(no verdict parsed; deferring to the gate)")
        self.notes.append(f"critic on {cand.version}: {'ACCEPT' if ok else 'REVERT'} ({cand_report.success_rate:.0%})")
        return ok, why
