"""T7: run a session of trials, collect the operator's labels, write the records.

A session is one skill version on one task over a list of seeds. Per trial:

1. `env.reset(seed)`: the arm homes, the seed draws a placement, the operator places the object.
2. The skill runs, wrapped in `GuardedSkill`, until the estimator reports success, the env stops,
   or the step budget runs out.
3. A `Labeller` gives the result: success, a failure tag, a note. On hardware that is a person;
   in a dry run it is the twin's true state. The estimator's opinion is kept separately.

Written under `out/`:
- `episodes.jsonl`: one `contracts.Episode` per trial; `success` is the label.
- `real.json`: the sidecar the contract has no room for: label source, the estimator's success,
  the VLM verdict (None until T6's judge exists), drawn and measured placement, safety events.
- `report.json`: the usual `EvalReport` (n/N with a Wilson interval) built from the labels.
- `arrays/episode_<seed>.npz`: T5's dataset arrays, plus commanded and measured joints for system ID.
- `video/episode_<seed>.gif` when the env can render and imageio is installed.

Approval: `Approvals` records which library versions a person has approved by name. A run on
hardware must go through `require_approved`; a dry run need not (run every version dry first).

Nothing in this file has run on a real arm: there is no ROS layer yet, so only dry runs exist.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, get_args

import numpy as np

from skillforge.contracts import Episode, EvalReport, FailureTag, Skill, SkillSource, TaskSpec
from skillforge.eval.harness import _classify, summarize, wilson

from .core import Measure, Placement, PlacementAborted, RealEnv
from .safety import GuardedSkill

FAILURE_TAGS: tuple[str, ...] = get_args(FailureTag)


# ---------------------------------------------------------------- approval


class NotApproved(PermissionError):
    pass


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


class Approvals:
    """Which skill versions a person has approved for hardware. One JSON file:
    {task_id: {version: {"code_sha", "by", "at"}}}. The sha pins the approval to the exact code."""

    def __init__(self, path: str | Path = "runs/real/approved.json"):
        self.path = Path(path)

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def approve(self, library, task_id: str, version: str, by: str) -> dict:
        """Approve a version that T3's library accepted. `by` is the person's name."""
        if not by.strip():
            raise ValueError("an approval needs the name of the person giving it")
        if not library.has(task_id, version):
            raise NotApproved(f"{task_id}@{version} is not in the library at {library.root}")
        if not library.meta(task_id, version)["accepted"]:
            raise NotApproved(f"{task_id}@{version} was evaluated but not accepted into the library")
        src, _ = library.load(task_id, version)
        entry = {"code_sha": code_sha(src.code), "by": by.strip(),
                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        data = self._read()
        data.setdefault(task_id, {})[version] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        return entry

    def require_approved(self, task_id: str, src: SkillSource) -> dict:
        entry = self._read().get(task_id, {}).get(src.version)
        if entry is None:
            raise NotApproved(f"{src.name}@{src.version} is not approved for hardware on {task_id}")
        if entry["code_sha"] != code_sha(src.code):
            raise NotApproved(f"{src.name}@{src.version}: the code differs from what {entry['by']} approved")
        return entry


# ---------------------------------------------------------------- operator and labels


@dataclass
class Label:
    success: bool
    failure: str | None = None  # a FailureTag, when not a success
    note: str = ""
    source: str = "operator"


class Labeller(Protocol):
    def label(self, env: RealEnv, auto_success: bool, events: list[str]) -> Label: ...


class TwinLabeller:
    """Dry runs: the label is the twin's TRUE sim state, which is what a person would see."""

    def label(self, env: RealEnv, auto_success: bool, events: list[str]) -> Label:
        return Label(success=bool(env.robot.true_success), source="sim_truth")


class TerminalLabeller:
    """Asks the person at the robot. `ask` and `say` are injectable for tests."""

    def __init__(self, ask: Callable[[str], str] = input, say: Callable[[str], None] = print):
        self.ask, self.say = ask, say

    def label(self, env: RealEnv, auto_success: bool, events: list[str]) -> Label:
        self.say(f"estimator says: {'success' if auto_success else 'no success'}")
        while (ans := self.ask("did it succeed? [y/n] ").strip().lower()) not in ("y", "n"):
            self.say("answer y or n")
        if ans == "y":
            return Label(success=True, note=self.ask("note (optional): ").strip())
        self.say("failure tags: " + ", ".join(FAILURE_TAGS))
        while (tag := self.ask("failure tag: ").strip()) not in FAILURE_TAGS:
            self.say("not a tag")
        return Label(success=False, failure=tag, note=self.ask("note (optional): ").strip())


class TerminalOperator:
    """Guides a person to place each object, checking the estimator's measurement of it.

    Enter = placed, measure it. 'f' = accept it as it is. 'a' = abort the trial.
    """

    def __init__(self, tolerance: float = 0.01, ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                 show: Callable[[int, dict[str, Placement]], None] | None = None):
        """`show(seed, targets)`, if given, is called before each prompt (e.g. to save a picture of
        the targets against the camera's current estimates)."""
        self.tolerance, self.ask, self.say, self.show = tolerance, ask, say, show

    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        for name, t in targets.items():
            self.say(f"seed {seed}: place '{name}' at x={t.pos[0]:.3f} y={t.pos[1]:.3f} m, "
                     f"turned {np.degrees(t.yaw_offset):+.0f} deg")
            while True:
                if self.show is not None:
                    self.show(seed, targets)
                ans = self.ask("Enter when placed, f to accept as is, a to abort: ").strip().lower()
                if ans == "a":
                    raise PlacementAborted(f"operator aborted placing '{name}' for seed {seed}")
                if ans == "f":
                    break
                est = measure().get(name)
                if est is None or not est.found:
                    self.say("the cameras cannot see it")
                    continue
                err = np.asarray(est.pos[:2]) - np.asarray(t.pos[:2])
                if np.linalg.norm(err) <= self.tolerance:
                    self.say(f"ok, {1e3 * np.linalg.norm(err):.0f} mm off")
                    break
                self.say(f"{1e3 * np.linalg.norm(err):.0f} mm off: move it {-1e3 * err[0]:+.0f} mm in x, {-1e3 * err[1]:+.0f} mm in y")


# ---------------------------------------------------------------- the session


@dataclass
class Trial:
    seed: int
    label_success: bool
    label_source: str
    label_failure: str | None
    note: str
    auto_success: bool  # the estimator's opinion; never the recorded result
    vlm_success: bool | None  # T6's judge, once it exists
    n_steps: int
    stopped: str | None  # why the env stopped the arm, if it did
    safety_events: list[tuple[int, str]]
    targets: dict[str, dict]
    measured: dict[str, float]
    arrays: str | None
    video: str | None


def _arrays(task: TaskSpec, obs_list, act_list, cmd_list, grip_list) -> dict[str, np.ndarray]:
    from skillforge.learning.features import action_vector, state_vector

    T = len(act_list)
    acts = np.full((T, 8), np.nan, dtype=np.float32)
    for i, a in enumerate(act_list):
        if a.mode == "ee":
            acts[i] = action_vector(a)
    extra: dict[str, np.ndarray] = {}
    try:
        extra["observation.state"] = np.array([state_vector(o, task) for o in obs_list[:T]],
                                              dtype=np.float32).reshape(T, -1)
    except AttributeError as e:
        # T5's features know single-object goals only (`goal_of` reads `success.object`); a
        # multi-object task (an AllOf, e.g. the can pyramid) has none. Keep everything else:
        # the commanded/measured arrays are what system ID needs.
        extra["observation.state_missing"] = np.array(f"T5 features do not support this task's success spec: {e}")
    return {
        **extra,
        "action": acts,
        "action.executed": acts.copy(),
        "label_valid": ~np.isnan(acts).any(axis=1),
        "timestamp": np.array([o.t for o in obs_list[:T]], dtype=np.float32),
        "frame_index": np.arange(T, dtype=np.int32),
        # for system ID: what was commanded at step t and what was measured after it
        "joint.start": np.asarray(obs_list[0].qpos, dtype=np.float64),
        "joint.command": np.array(cmd_list, dtype=np.float64).reshape(T, -1),
        "joint.measured": np.array([o.qpos for o in obs_list[1:]], dtype=np.float64).reshape(T, -1),
        "gripper.command": np.array(grip_list, dtype=np.float32),
        "gripper.width": np.array([o.gripper_width for o in obs_list[1:]], dtype=np.float32),
    }


def run_trial(env: RealEnv, skill: Skill, task: TaskSpec, seed: int, labeller: Labeller, out: Path | None = None,
              video_every: int = 4, judge=None) -> tuple[Episode, Trial]:
    """`judge`: T6's `VLMJudge` (or anything with `judge(task, images) -> verdict`). Each call is
    billed, so it is off unless one is passed. Its verdict is recorded and never used as the result."""
    t0 = time.perf_counter()
    guarded = GuardedSkill(skill, env)
    obs = env.reset(seed)
    guarded.reset(task)
    obs_list, act_list, cmd_list, grip_list, frames = [obs], [], [], [], []
    events: list[str] = []
    auto = terminated = False
    crash = ""
    n = 0
    try:
        for n in range(1, task.max_steps + 1):  # noqa: B007 (n is read after the loop)
            action = guarded.act(obs)
            obs, info = env.step(action)
            act_list.append(action)
            q, g = env.command
            cmd_list.append(q)
            grip_list.append(g)
            obs_list.append(obs)
            events += info.events
            if out is not None and n % video_every == 0 and (img := env.render()) is not None:
                frames.append(img)
            if info.success or info.terminated:
                auto, terminated = info.success, info.terminated
                break
    except Exception as e:  # GuardedSkill has already stopped the arm
        crash = f"{type(e).__name__}: {e}"
        n = len(act_list)
    vlm = None
    if judge is not None and (final := env.render()) is not None:
        vlm = judge.judge(task, [final]).success
    label = labeller.label(env, auto, events)
    failure, detail = None, label.note
    if not label.success:
        tag, why = ("exception", crash) if crash else _classify(events, terminated, timed_out=n >= task.max_steps)
        failure = label.failure or tag
        detail = "; ".join(x for x in (label.note, f"auto: {tag}" + (f" ({why})" if why else "")) if x)
    arrays = video = None
    if out is not None:
        if act_list:
            p = out / "arrays" / f"episode_{seed}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **_arrays(task, obs_list, act_list, cmd_list, grip_list))
            arrays = str(p.relative_to(out))
        if frames:
            try:
                from skillforge.eval.video import write_gif

                write_gif(frames, out / "video" / f"episode_{seed}.gif", fps=task.control_hz / video_every)
                video = f"video/episode_{seed}.gif"
            except ImportError:
                pass
    sample = env.randomization_sample()
    episode = Episode(
        task_id=task.id, skill=skill.name, skill_version=skill.version, seed=seed, randomization=sample,
        n_steps=n, success=label.success, failure=failure, failure_detail=detail, events=events,
        duration_s=time.perf_counter() - t0, arrays=arrays, video=video,
    )
    trial = Trial(
        seed=seed, label_success=label.success, label_source=label.source, label_failure=failure, note=label.note,
        auto_success=bool(auto), vlm_success=vlm, n_steps=n, stopped=env.stopped, safety_events=env.safety_log,
        targets={k: asdict(v) for k, v in env.targets.items()},
        measured={k: v for k, v in sample.items() if ".placed_" in k or k.endswith(".place_error")},
        arrays=arrays, video=video,
    )
    return episode, trial


def agreement(trials: list[Trial]) -> dict:
    """How often the estimator's success call matched the label."""
    n = len(trials)
    same = sum(t.auto_success == t.label_success for t in trials)
    return {
        "n": n,
        "agree": same,
        "auto_only": sum(t.auto_success and not t.label_success for t in trials),  # false successes
        "label_only": sum(t.label_success and not t.auto_success for t in trials),  # missed successes
        "rate": same / n if n else None,
        "ci95": list(wilson(same, n)),
    }


def run_session(
    env: RealEnv,
    skill: Skill,
    task: TaskSpec,
    seeds: Iterable[int],
    labeller: Labeller,
    out: str | Path,
    *,
    mode: str = "dry_run",  # "dry_run" or "hardware"
    judge=None,
    say: Callable[[str], None] = print,
) -> tuple[EvalReport, list[Trial]]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes, trials, aborted = [], [], []
    with (out / "episodes.jsonl").open("w") as f:
        for seed in seeds:
            try:
                ep, trial = run_trial(env, skill, task, int(seed), labeller, out, judge=judge)
            except PlacementAborted as e:
                aborted.append(int(seed))
                say(f"seed {seed}: skipped ({e})")
                continue
            f.write(ep.model_dump_json() + "\n")
            f.flush()
            episodes.append(ep)
            trials.append(trial)
            say(f"seed {seed}: {'success' if ep.success else ep.failure} in {ep.n_steps} steps"
                + (f", safety: {sorted({e for _, e in trial.safety_events})}" if trial.safety_events else ""))
    report = summarize(task, skill.name, skill.version, episodes)
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    sidecar = {
        "mode": mode,
        "task_id": task.id,
        "skill": skill.name,
        "skill_version": skill.version,
        "limits": asdict(env.safety.limits),
        "workspace": asdict(env.safety.workspace),
        "aborted_seeds": aborted,
        "auto_vs_label": agreement(trials),
        "trials": [asdict(t) for t in trials],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "real.json").write_text(json.dumps(sidecar, indent=2))
    say(report.summary())
    return report, trials
