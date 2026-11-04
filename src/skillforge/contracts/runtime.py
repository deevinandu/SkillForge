"""Runtime contracts: what flows between the env, a skill and the eval harness each step.

Observation/Action are plain dataclasses over numpy arrays (hot path, not serialized
directly). Episode and EvalReport are pydantic models (written to disk, read by agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .specs import TaskSpec

# ---------------------------------------------------------------- per-step data


@dataclass
class ObjectState:
    pos: np.ndarray  # (3,) world frame, observation noise already applied
    quat: np.ndarray  # (4,) w, x, y, z


@dataclass
class Observation:
    t: float  # seconds since reset
    qpos: np.ndarray  # arm joint positions
    qvel: np.ndarray
    ee_pos: np.ndarray  # (3,)
    ee_quat: np.ndarray  # (4,)
    gripper_width: float  # metres, 0 = closed
    objects: dict[str, ObjectState] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)  # camera name -> HxWx3 uint8
    contact_force: np.ndarray | None = None  # (3,) at the EE, if the env provides it


@dataclass
class Action:
    """A target for the env's low-level controller.

    mode 'ee': move the end effector toward (ee_pos, ee_quat); the env's controller does IK.
    mode 'joint': track joint targets directly.
    gripper: 0 = fully open, 1 = fully closed.
    """

    mode: Literal["ee", "joint"] = "ee"
    ee_pos: np.ndarray | None = None
    ee_quat: np.ndarray | None = None
    joint_targets: np.ndarray | None = None
    gripper: float = 0.0

    def __post_init__(self) -> None:
        if self.mode == "ee" and self.ee_pos is None:
            raise ValueError("Action(mode='ee') needs ee_pos")
        if self.mode == "joint" and self.joint_targets is None:
            raise ValueError("Action(mode='joint') needs joint_targets")
        if not 0.0 <= self.gripper <= 1.0:
            raise ValueError(f"gripper must be in [0, 1], got {self.gripper}")


@dataclass
class StepInfo:
    success: bool
    terminated: bool  # task can no longer succeed (e.g. object fell off the table)
    events: list[str] = field(default_factory=list)  # e.g. 'collision:arm-table', 'grasp_slip'


# ---------------------------------------------------------------- interfaces


@runtime_checkable
class SimEnv(Protocol):
    """Owned by T1. One env instance = one TaskSpec's scene."""

    task: TaskSpec

    def reset(self, seed: int) -> Observation:
        """Rebuild state, sample randomization from task.scene.randomization with this seed."""

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        """Advance one control period (1 / task.control_hz seconds)."""

    def randomization_sample(self) -> dict[str, float]:
        """The values drawn at the last reset, for logging and failure analysis."""

    def render(self, camera: str | None = None) -> np.ndarray | None:
        """HxWx3 uint8 frame, or None if rendering is unavailable."""

    def close(self) -> None: ...


@runtime_checkable
class Skill(Protocol):
    """Owned by T2 (hand-written) and T3 (agent-written). A closed-loop policy for one task.

    Skills must be deterministic given (task, observations): randomness comes only from the env seed.
    """

    name: str
    version: str

    def reset(self, task: TaskSpec) -> None: ...

    def act(self, obs: Observation) -> Action: ...


# ---------------------------------------------------------------- records


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


FailureTag = Literal[
    "timeout",
    "grasp_miss",
    "grasp_slip",
    "collision",
    "object_dropped",
    "wrong_placement",
    "ik_failure",
    "exception",
    "terminated",
    "other",
]


class Episode(Record):
    """One rollout. Arrays live in a sibling .npz (path in `arrays`), not inline."""

    task_id: str
    skill: str
    skill_version: str
    seed: int
    randomization: dict[str, float]
    n_steps: int
    success: bool
    failure: FailureTag | None = None
    failure_detail: str = ""
    events: list[str] = Field(default_factory=list)
    duration_s: float
    arrays: str | None = None  # path to .npz with obs/action time series, for learning (T5)
    video: str | None = None


class FailureCase(Record):
    seed: int
    tag: FailureTag
    detail: str
    randomization: dict[str, float]
    video: str | None = None


class EvalReport(Record):
    """Owned by T4. The thing agents (T3) read to decide what to fix next."""

    task_id: str
    skill: str
    skill_version: str
    n_rollouts: int
    n_success: int
    success_rate: float
    ci95: tuple[float, float]  # Wilson interval
    failure_counts: dict[str, int]
    failures: list[FailureCase]  # a sample, worst-first
    created_at: str  # ISO 8601

    def summary(self) -> str:
        lo, hi = self.ci95
        fails = ", ".join(f"{k}={v}" for k, v in sorted(self.failure_counts.items(), key=lambda kv: -kv[1]))
        return (
            f"{self.skill}@{self.skill_version} on {self.task_id}: "
            f"{self.n_success}/{self.n_rollouts} = {self.success_rate:.0%} "
            f"(95% CI {lo:.0%}-{hi:.0%}); failures: {fails or 'none'}"
        )
