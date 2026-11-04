"""A deterministic, offline improver that applies scripted edits. For tests and dry runs.

Each scripted edit is one of:
- a str: the complete new file,
- a (old, new) tuple: a literal replacement in the current code (all occurrences),
- a callable(code) -> code.
After the script runs out it keeps returning the current skill unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from skillforge.contracts import EvalReport, SkillSource, TaskSpec

from .budget import UsageMeter
from .library import bump_version

Edit = str | tuple[str, str] | Callable[[str], str]

# Used by `skillforge forge --improver fake`: small, harmless tweaks to the scripted PickPlace.
DEFAULT_SCRIPT: tuple[Edit, ...] = (
    ("tol = 0.008", "tol = 0.006"),
    ("settle_steps = 4", "settle_steps = 5"),
    ("hover = 0.10", "hover = 0.08"),
)


class FakeImprover:
    def __init__(self, script: Sequence[Edit] = DEFAULT_SCRIPT, rationale: str = "scripted edit"):
        self.script = list(script)
        self.rationale = rationale
        self.calls = 0
        self.usage = UsageMeter()  # always zero: no network

    def propose(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> SkillSource:
        i = self.calls
        self.calls += 1
        if i >= len(self.script):
            return current
        edit = self.script[i]
        if isinstance(edit, str):
            code = edit
        elif isinstance(edit, tuple):
            code = current.code.replace(edit[0], edit[1])
        else:
            code = edit(current.code)
        top = max(report.failure_counts, key=report.failure_counts.get, default="none")
        return SkillSource(
            name=current.name,
            version=bump_version(current.version),
            code=code,
            entrypoint=current.entrypoint,
            rationale=f"{self.rationale} #{i + 1} (top failure: {top})",
            parent_version=current.version,
        )
