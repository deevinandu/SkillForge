"""`LearnedSkill`: a trained BC checkpoint behind the `contracts.Skill` interface.

It runs through the same `evaluate()` / `evaluate_parallel()` harness as scripted skills. For
the parallel harness (which ships skills as code) use `learned_source(checkpoint)`, a small
`SkillSource` that subclasses `LearnedSkill` with the checkpoint path baked in.

Execution: every step the policy predicts a chunk of `horizon` future actions; predictions
are stored in world frame (absolute ee targets) and the action for the current step is the
weighted mean of every stored prediction that covers it, weight exp(-m * i) with i = 0 for the
oldest (ACT's temporal ensembling; m = `ensemble_m`, 0 = uniform). `ensemble_m=None` disables
ensembling and executes only the newest prediction's first action.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from skillforge.contracts import Action, Observation, SkillSource, TaskSpec

from .features import Layout, policy_features, rot6_to_quat, state_vector
from .policy import LoadedPolicy


class LearnedSkill:
    name = "learned"
    version = "0"

    def __init__(
        self,
        checkpoint: str | Path,
        ensemble_m: float | None = 0.1,
        gripper_threshold: float | None = 0.5,
        threads: int = 1,
    ):
        self.policy = LoadedPolicy(checkpoint, threads=threads)
        meta = self.policy.meta
        self.name = f"learned_{meta['teacher']['name']}"
        self.version = f"bc-{self.policy.sha}"
        self.ensemble_m = ensemble_m
        self.gripper_threshold = gripper_threshold
        self.layout = Layout(meta["state_names"], meta["object"])
        self.task: TaskSpec | None = None

    def reset(self, task: TaskSpec) -> None:
        meta = self.policy.meta
        if task.id != meta["task_id"]:
            raise ValueError(f"checkpoint was trained on task '{meta['task_id']}', not '{task.id}'")
        self.task = task
        self.t = 0
        self.hist: deque[np.ndarray] = deque(maxlen=self.policy.n_hist)
        self.preds: dict[int, list[np.ndarray]] = {}  # step -> predictions (oldest first), 10-vectors in world frame

    def act(self, obs: Observation) -> Action:
        assert self.task is not None, "reset() first"
        s = state_vector(obs, self.task)
        if len(s) != len(self.policy.meta["state_names"]):
            raise ValueError("observation layout does not match the checkpoint's")
        f = policy_features(s[None], self.layout)[0]
        if not self.hist:
            for _ in range(self.policy.n_hist):
                self.hist.append(f)
        else:
            self.hist.append(f)
        y = self.policy.predict(np.concatenate(list(self.hist)))  # (H, 10), pos relative to ee now
        ee = obs.ee_pos
        y = y.copy()
        y[:, :3] += ee
        t = self.t
        if self.ensemble_m is None:
            cur = y[0]
        else:
            for k in range(len(y)):
                self.preds.setdefault(t + k, []).append(y[k])
            ps = np.stack(self.preds.pop(t))
            w = np.exp(-self.ensemble_m * np.arange(len(ps)))
            cur = (w[:, None] * ps).sum(0) / w.sum()
        self.t += 1
        d = cur[:3] - ee
        n = np.linalg.norm(d)
        if n > self.policy.pos_clip:  # same clip the training targets saw
            d *= self.policy.pos_clip / n
        g = float(np.clip(cur[9], 0.0, 1.0))
        if self.gripper_threshold is not None:
            g = 1.0 if g >= self.gripper_threshold else 0.0
        return Action(mode="ee", ee_pos=ee + d, ee_quat=rot6_to_quat(cur[3:9]), gripper=g)


def learned_source(checkpoint: str | Path, **overrides) -> SkillSource:
    """A SkillSource for the parallel harness: subclass with the checkpoint path baked in."""
    path = str(Path(checkpoint).resolve())
    probe = LearnedSkill(path, **overrides)
    code = (
        "from skillforge.learning.skill import LearnedSkill as _Base\n\n\n"
        "class Learned(_Base):\n"
        f"    name = {probe.name!r}\n"
        f"    version = {probe.version!r}\n\n"
        "    def __init__(self):\n"
        f"        super().__init__({path!r}, **{overrides!r})\n"
    )
    return SkillSource(name=probe.name, version=probe.version, code=code, entrypoint="Learned",
                       rationale=f"behavior cloning of {probe.policy.meta['teacher']} from {probe.policy.meta['datasets']}")


__all__ = ["LearnedSkill", "learned_source"]
