"""The do-nothing improver: returns the skill unchanged. Lets the forge loop run end to end in tests."""

from __future__ import annotations

from skillforge.contracts import EvalReport, SkillSource, TaskSpec


class NullImprover:
    def propose(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> SkillSource:
        return current
