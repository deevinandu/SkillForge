"""T6: the VLM success judge, against a fake client. No network, no spend."""

import base64
import json
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from skillforge.agents.budget import BudgetExceeded
from skillforge.perception.vlm_judge import (
    FALLBACK_BETA,
    Confusion,
    JudgeError,
    VLMJudge,
    criterion,
    png_bytes,
    score,
)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    """client.beta.messages.stream(...).get_final_message(), replying from a script."""

    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.calls, self.stop_reason = list(replies), [], stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        usage = SimpleNamespace(input_tokens=1500, output_tokens=120, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
        return _Stream(SimpleNamespace(content=content, usage=usage, stop_reason=self.stop_reason, model=kw["model"]))


def image(value=120, size=(48, 64)):
    return np.full((*size, 3), value, dtype=np.uint8)


def yes(reason="the cube is in the bin"):
    return {"reason": reason, "success": True}


def no(reason="the cube is on the rim"):
    return {"reason": reason, "success": False}


def test_png_encoding_round_trips():
    img = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    png = png_bytes(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    idat = png[png.index(b"IDAT") + 4 : png.index(b"IEND") - 8]
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(5, 1 + 7 * 3)
    assert (rows[:, 0] == 0).all() and (rows[:, 1:].reshape(5, 7, 3) == img).all()
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    assert (np.asarray(Image.open(io.BytesIO(png))) == img).all()
    with pytest.raises(ValueError):
        png_bytes(np.zeros((4, 4)))


def test_judge_sends_the_images_and_the_task_and_parses_the_verdict(stub_task):
    client = FakeClient([yes()])
    judge = VLMJudge(client)
    v = judge.judge(stub_task, [image(10), image(200)])
    assert v.success is True and v.reason == "the cube is in the bin" and v.model == "claude-sonnet-5-5" and v.usd > 0
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "low" and call["output_config"]["format"]["type"] == "json_schema"
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert "tool_choice" not in call and "temperature" not in call
    blocks = call["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"]
    assert base64.standard_b64decode(blocks[0]["source"]["data"])[:4] == b"\x89PNG"
    assert stub_task.instruction in blocks[2]["text"] and "'cube' is resting inside the 'bin'" in blocks[2]["text"]
    assert "rim" in call["system"]


def test_criterion_reads_the_task(stub_task):
    assert criterion(stub_task) == "the 'cube' is resting inside the 'bin' (a 10 x 10 cm area on the table), released by the gripper"


def test_no_usable_answer_is_not_a_verdict(stub_task):
    assert VLMJudge(FakeClient(["not json"])).judge(stub_task, [image()]).success is None
    assert VLMJudge(FakeClient([yes()], stop_reason="refusal")).judge(stub_task, [image()]).success is None
    assert VLMJudge(FakeClient([{"reason": "x"}])).judge(stub_task, [image()]).success is None
    with pytest.raises(JudgeError):
        VLMJudge(FakeClient([])).judge(stub_task, [])


def test_the_budget_cap_refuses_a_call_before_it_is_made(stub_task):
    client = FakeClient([yes(), yes()])
    judge = VLMJudge(client, budget_usd=0.022)  # worst case of one call is about $0.02
    judge.judge(stub_task, [image()])
    with pytest.raises(BudgetExceeded):
        judge.judge(stub_task, [image()])
    assert len(client.calls) == 1 and judge.usage.usd < 0.022


def test_score_is_a_confusion_matrix_against_true_success(stub_task):
    client = FakeClient([yes(), yes(), no(), no(), "garbage"])
    cases = [([image()], True), ([image()], False), ([image()], False), ([image()], True), ([image()], True)]
    seen = []
    c = score(VLMJudge(client), stub_task, cases, on_case=lambda v, truth: seen.append((v.success, truth)))
    assert (c.true_success, c.false_success, c.true_failure, c.missed_success, c.declined) == (1, 1, 1, 1, 1)
    assert c.accuracy == 0.5 and c.usd > 0 and len(seen) == 5
    assert "false success 1" in c.summary()
    assert Confusion.model_validate_json(c.model_dump_json()) == c


def test_score_stops_cleanly_when_the_budget_runs_out(stub_task):
    client = FakeClient([yes()] * 10)
    c = score(VLMJudge(client, budget_usd=0.05), stub_task, [([image()], True)] * 10)
    assert 0 < c.true_success < 10 and len(client.calls) == c.true_success


def test_final_frames_reports_true_success_not_the_judges(stub_task):
    from skillforge.eval.bench import RenderStubEnv
    from skillforge.perception.vlm_judge import final_frames
    from skillforge.skills import PickPlace

    frames, truth = final_frames(RenderStubEnv(stub_task), PickPlace(), stub_task, 0)
    assert truth is True and len(frames) == 1 and frames[0].ndim == 3
