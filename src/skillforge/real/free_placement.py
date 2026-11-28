"""T7: free placement. The person puts the objects anywhere in a start area; the camera measures
where they are; the layout is checked; the sim predicts the skill's success on THAT layout; then
the real trial runs.

This replaces "put each object at the seed's drawn spot" for the lab cell. The skill never needed
fixed spots (it reads object poses from the camera); the spots existed so a real trial could be
paired with a sim trial. Here the pairing is better: the sim runs the exact measured layout,
`n` times, with the task's physical randomization (friction, mass, gains, latency, observation
noise) and a few mm of start jitter for the camera's measurement error. That gives a prediction
for this trial, recorded next to the person's label.

- `check_layout`: every object seen, inside the start area, far enough apart for the fingers.
- `pinned_task`: the task with each object's start at its measured pose and the xy jitter cut to
  the measurement error (a TaskSpec, saved with the session, replayable later).
- `FreePlacementOperator`: the `Operator` that runs measure -> check -> predict -> confirm.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skillforge.contracts import SkillSource, TaskSpec
from skillforge.perception import Estimate

from .core import Measure, Placement, PlacementAborted


@dataclass(frozen=True)
class StartArea:
    """Where objects may start (world frame, metres). The lab default keeps every can inside the
    calibrated camera view and the arm's comfortable reach, clear of franka-interface's walls
    (front x 0.75, sides +-0.47) and of the pyramid's spots (y 0.08 +- 0.03)."""

    x: tuple[float, float] = (0.30, 0.62)
    y: tuple[float, float] = (-0.30, -0.02)

    def contains(self, p) -> bool:
        return self.x[0] <= p[0] <= self.x[1] and self.y[0] <= p[1] <= self.y[1]


LAB_START_AREA = StartArea()


def check_layout(task: TaskSpec, estimates: dict[str, Estimate], area: StartArea = LAB_START_AREA,
                 finger_gap: float = 0.04, min_confidence: float = 0.5) -> list[str]:
    """What is wrong with this layout, one sentence each; empty if it is fine to run.

    finger_gap: free space (m) needed between two objects' sides for the open fingers to pass.
    """
    problems = []
    movable = [o for o in task.scene.objects if not o.fixed]
    seen = {n: e for n, e in estimates.items() if e.found}
    if len(seen) < len(movable):
        problems.append(f"the camera sees {len(seen)} of {len(movable)} objects: put all of them in view, standing up")
    for n, e in seen.items():
        if not area.contains(e.pos):
            problems.append(f"{n} at x={e.pos[0]:.2f} y={e.pos[1]:.2f} is outside the start area "
                            f"(x {area.x[0]:.2f}..{area.x[1]:.2f}, y {area.y[0]:.2f}..{area.y[1]:.2f})")
        if e.confidence < min_confidence:
            problems.append(f"{n} at x={e.pos[0]:.2f} y={e.pos[1]:.2f} is not clearly seen "
                            f"(fit {e.confidence:.2f}): is it upright and not hidden behind another?")
    names = sorted(seen)
    radius = {o.name: float(o.size[0]) for o in movable}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = float(np.linalg.norm(seen[a].pos[:2] - seen[b].pos[:2])) - radius.get(a, 0) - radius.get(b, 0)
            if gap < finger_gap:
                problems.append(f"{a} and {b} are {max(gap, 0) * 100:.1f} cm apart; leave at least "
                                f"{finger_gap * 100:.0f} cm between them for the fingers")
    return problems


def pinned_task(task: TaskSpec, positions: dict[str, np.ndarray], xy_jitter: float = 0.003) -> TaskSpec:
    """`task` with each named object starting at its measured position (z from the task, the
    object resting on the table), and the start jitter cut to `xy_jitter` (the camera's error).
    Everything physical keeps the task's randomization."""
    data = task.model_dump()
    data["id"] = f"{task.id}__layout"
    for o in data["scene"]["objects"]:
        if o["name"] in positions:
            p = positions[o["name"]]
            o["pose"]["pos"] = [round(float(p[0]), 4), round(float(p[1]), 4), o["pose"]["pos"][2]]
    data["scene"]["randomization"]["object_xy_jitter"] = {"lo": 0.0, "hi": xy_jitter}
    return TaskSpec.model_validate(data)


@dataclass
class Prediction:
    n_success: int
    n: int
    lo: float
    hi: float
    failures: dict[str, int]

    def line(self) -> str:
        fails = (", failures " + ", ".join(f"{k} x{v}" for k, v in self.failures.items())) if self.failures else ""
        return f"sim predicts {self.n_success}/{self.n} for this layout (95% CI {self.lo:.0%}-{self.hi:.0%}){fails}"


def sim_prediction(task: TaskSpec, src: SkillSource, n: int = 50, workers: int | None = None) -> Prediction:
    """Run the skill `n` times in sim on `task` (normally a `pinned_task`)."""
    from skillforge.eval import evaluate_parallel
    from skillforge.eval.harness import wilson
    from skillforge.sim import make_env

    report, eps = evaluate_parallel(make_env, src, task, range(n), workers=workers)
    lo, hi = wilson(report.n_success, report.n_rollouts)
    fails: dict[str, int] = {}
    for e in eps:
        if not e.success:
            fails[str(e.failure)] = fails.get(str(e.failure), 0) + 1
    return Prediction(report.n_success, report.n_rollouts, lo, hi, fails)


@dataclass
class FreePlacementOperator:
    """`Operator` for free placement. Ignores the seed's drawn targets.

    Enter: measure the table and check the layout. When it is fine, the sim predicts it, and
    Enter runs the trial; 'r' re-measures (after moving something); 'a' aborts the trial.
    `predict(task_with_layout)` and `show()` are injected (the CLI passes the sim and a picture).
    """

    task: TaskSpec
    area: StartArea = LAB_START_AREA
    predict: Callable[[TaskSpec], Prediction] | None = None
    show: Callable[[], None] | None = None
    min_prediction: float = 0.8  # warn below this predicted success rate
    out: Path | None = None  # saves layout_<seed>.json (pinned task + prediction) here
    ask: Callable[[str], str] = input
    say: Callable[[str], None] = print
    records: dict[int, dict] = field(default_factory=dict)

    def place(self, seed: int, targets: dict[str, Placement], measure: Measure) -> None:
        n = sum(not o.fixed for o in self.task.scene.objects)
        self.say(f"\ntrial {seed}: put the {n} objects anywhere in the start area "
                 f"(x {self.area.x[0]:.2f}..{self.area.x[1]:.2f} m out from the base, "
                 f"y {self.area.y[0]:.2f}..{self.area.y[1]:.2f} m), standing up, spread apart.")
        while True:
            if self.ask("Enter when placed (a = abort): ").strip().lower() == "a":
                raise PlacementAborted(f"operator aborted trial {seed}")
            if self.show is not None:
                self.show()
            est = measure()
            problems = check_layout(self.task, est, self.area)
            if problems:
                self.say("not yet:\n  - " + "\n  - ".join(problems))
                continue
            positions = {k: e.pos.copy() for k, e in est.items() if e.found}
            pinned = pinned_task(self.task, positions)
            record = {"seed": seed, "positions": {k: [round(float(v), 4) for v in p] for k, p in positions.items()},
                      "pinned_task": json.loads(pinned.model_dump_json())}
            if self.predict is not None:
                self.say("layout ok; running the sim on it...")
                pred = self.predict(pinned)
                record["prediction"] = {"n_success": pred.n_success, "n": pred.n, "ci95": [pred.lo, pred.hi],
                                        "failures": pred.failures}
                self.say(pred.line())
                if pred.n and pred.n_success / pred.n < self.min_prediction:
                    self.say(f"that is below {self.min_prediction:.0%}: consider spreading the objects out more")
            ans = self.ask("Enter = run the trial, r = re-measure, a = abort: ").strip().lower()
            if ans == "a":
                raise PlacementAborted(f"operator aborted trial {seed}")
            if ans == "r":
                continue
            self.records[seed] = record
            if self.out is not None:
                self.out.mkdir(parents=True, exist_ok=True)
                (self.out / f"layout_{seed}.json").write_text(json.dumps(record, indent=2))
            return
