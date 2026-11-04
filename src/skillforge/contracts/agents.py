"""Agent contract (T3): turn an eval report into a better skill."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .runtime import EvalReport
from .specs import TaskSpec


class SkillSource(BaseModel):
    """A skill as code. `entrypoint` is the class name inside `code` implementing the Skill protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    code: str
    entrypoint: str
    rationale: str = ""  # why this version differs from its parent, written by whoever made it
    parent_version: str | None = None


@runtime_checkable
class SkillImprover(Protocol):
    def propose(self, task: TaskSpec, current: SkillSource, report: EvalReport) -> SkillSource:
        """Return a new version of the skill aimed at the failures in `report`."""


@runtime_checkable
class SkillCritic(Protocol):
    """Optional second hook: veto a candidate after it is evaluated. Improvers may implement both."""

    def review(
        self,
        task: TaskSpec,
        parent: SkillSource,
        parent_report: EvalReport,
        candidate: SkillSource,
        candidate_report: EvalReport,
    ) -> tuple[bool, str]:
        """Return (accept, reason)."""
