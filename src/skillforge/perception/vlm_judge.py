"""T6: a success judge for the real robot, where no true state exists.

`VLMJudge.judge(task, images)` sends the final camera frame(s) and the task to Claude and gets
back `{success, reason}`. On hardware it is a second opinion next to the operator's label and
the estimator's; it never decides a result. In sim it is only ever SCORED, as a classifier
against true success (`score`), and never decides success there either.

Cost: every call is billed. The judge takes a `budget_usd` cap and refuses a call whose worst
case would pass it (`agents.budget.BudgetExceeded`), and logs the cost of each call.

**Status: built and tested against a fake client only. No real API call has been made, so its
accuracy is unmeasured.** The confusion matrix over 200+ sim episodes the brief asks for needs
paid calls.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from skillforge.agents.budget import UsageMeter, price_for
from skillforge.contracts import ObjectInRegion, TaskSpec
from skillforge.eval.harness import wilson

DEFAULT_MODEL = "claude-sonnet-5-5"  # `claude-opus-5-5` is the stronger, dearer option
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 2000

SYSTEM = """You judge whether a robot arm finished a tabletop manipulation task, from camera images taken after it stopped.

Judge only what the images show. Be strict:
- An object resting on the rim or edge of its goal, leaning out of it, or still held by the gripper is NOT a success.
- If the object or the goal cannot be seen well enough to tell, answer false and say what is hidden.
Give the reason in one or two sentences, naming what you saw."""

SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "success": {"type": "boolean"}},
    "required": ["reason", "success"],
    "additionalProperties": False,
}


class JudgeError(RuntimeError):
    pass


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 image as PNG with the standard library only."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 image, got shape {a.shape}")
    a = np.ascontiguousarray(a[..., :3].astype(np.uint8))
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), dtype=np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()  # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def criterion(task: TaskSpec) -> str:
    """The task's success condition in words, for the prompt."""
    s = task.success
    if isinstance(s, ObjectInRegion):
        r = task.scene.region(s.region)
        size = " x ".join(f"{200 * h:.0f}" for h in r.half_extents[:2])
        return (f"the '{s.object}' is resting inside the '{s.region}' (a {size} cm area on the table), "
                f"released by the gripper")
    return f"the '{s.object}' is held at least {100 * s.min_height:.0f} cm above where it started"


@dataclass
class Verdict:
    success: bool | None  # None: the judge gave no usable answer (declined, truncated, malformed)
    reason: str
    model: str
    usd: float


class VLMJudge:
    """client: an `anthropic.Anthropic`-like object; built from the environment if omitted
    (needs `ANTHROPIC_API_KEY` and the `agents` extra). Tests pass a fake."""

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        budget_usd: float | None = None,
        effort: str = "low",
        use_fallbacks: bool = True,
        max_images: int = 4,
    ):
        if client is None:
            import anthropic  # lazy: the `agents` extra

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.max_images = max_images
        self.usage = UsageMeter(budget_usd=budget_usd)

    def _worst_case(self, images: Sequence[np.ndarray], text_chars: int) -> float:
        # An image costs about width * height / 750 tokens; text about 1 token per 3 characters.
        tokens = sum(i.shape[0] * i.shape[1] / 750 for i in images) + text_chars / 3
        p = price_for(self.model)
        return (tokens * p.input + MAX_TOKENS * p.output) / 1e6

    def judge(self, task: TaskSpec, images: Sequence[np.ndarray]) -> Verdict:
        images = [np.asarray(i) for i in images][: self.max_images]
        if not images:
            raise JudgeError("the judge needs at least one image")
        text = (f"Task given to the robot: {task.instruction}\n"
                f"It counts as done when {criterion(task)}.\n"
                f"The {len(images)} image(s) above show the scene after the robot stopped. Was the task done?")
        self.usage.check(self._worst_case(images, len(SYSTEM) + len(text)), "judge")  # raises BudgetExceeded
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png_bytes(i)).decode("ascii")}}
            for i in images
        ]
        content.append({"type": "text", "text": text})
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if self.use_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        with self.client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        call = self.usage.record("judge", self.model, getattr(msg, "model", None), msg.usage)
        if msg.stop_reason in ("refusal", "max_tokens"):
            return Verdict(None, f"no verdict: stop_reason {msg.stop_reason}", call.model, call.usd)
        reply = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            data = json.loads(reply)
            return Verdict(bool(data["success"]), str(data["reason"]), call.model, call.usd)
        except (ValueError, KeyError, TypeError):
            return Verdict(None, f"no verdict: unparseable reply {reply[:80]!r}", call.model, call.usd)


# ---------------------------------------------------------------- scoring against truth


class Confusion(BaseModel):
    """The judge as a classifier of true success. `declined` cases gave no verdict."""

    model_config = ConfigDict(extra="forbid")

    true_success: int  # said success, was success
    false_success: int  # said success, was not: the dangerous kind on hardware
    true_failure: int
    missed_success: int
    declined: int
    accuracy: float | None
    ci95: tuple[float, float]
    usd: float

    def summary(self) -> str:
        n = self.true_success + self.false_success + self.true_failure + self.missed_success
        acc = "n/a" if self.accuracy is None else f"{self.accuracy:.0%} ({self.ci95[0]:.0%}-{self.ci95[1]:.0%})"
        return (f"judge vs truth on {n} cases: accuracy {acc}; false success {self.false_success}, "
                f"missed success {self.missed_success}, no verdict {self.declined}; ${self.usd:.4f}")


def score(judge: VLMJudge, task: TaskSpec, cases: Iterable[tuple[Sequence[np.ndarray], bool]],
          on_case: Callable[[Verdict, bool], None] | None = None) -> Confusion:
    """`cases`: (final images, TRUE success). Stops cleanly when the budget runs out."""
    from skillforge.agents.budget import BudgetExceeded

    c = {"ts": 0, "fs": 0, "tf": 0, "ms": 0, "declined": 0}
    for images, truth in cases:
        try:
            v = judge.judge(task, images)
        except BudgetExceeded:
            break
        if on_case is not None:
            on_case(v, truth)
        if v.success is None:
            c["declined"] += 1
        else:
            c[{(True, True): "ts", (True, False): "fs", (False, False): "tf", (False, True): "ms"}[(v.success, truth)]] += 1
    n, right = c["ts"] + c["fs"] + c["tf"] + c["ms"], c["ts"] + c["tf"]
    return Confusion(true_success=c["ts"], false_success=c["fs"], true_failure=c["tf"], missed_success=c["ms"],
                     declined=c["declined"], accuracy=right / n if n else None, ci95=wilson(right, n), usd=judge.usage.usd)


def final_frames(env, skill, task: TaskSpec, seed: int, cameras: Sequence[str | None] = (None,)) -> tuple[list[np.ndarray], bool]:
    """Run one sim episode and return its last frame from each camera with the TRUE outcome.
    Failures that end near the goal (on the rim, dropped beside it) are the near-misses the judge
    must get right; they arise on their own under the task's randomization."""
    obs = env.reset(seed)
    skill.reset(task)
    success = False
    for _ in range(task.max_steps):
        obs, info = env.step(skill.act(obs))
        if info.success or info.terminated:
            success = info.success
            break
    frames = [f for f in (env.render(c) for c in cameras) if f is not None]
    return frames, bool(success)
