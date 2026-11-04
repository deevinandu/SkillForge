"""ClaudeImprover against a scripted stand-in client. No network; the real API is NOT exercised."""

from types import SimpleNamespace

import pytest
from _skills import BROKEN, GOOD, PARTIAL, src

from skillforge.agents import (
    BudgetExceeded,
    ClaudeImprover,
    ImproverError,
    MissingCredentials,
    run_loop,
)
from skillforge.agents.budget import UsageMeter, price_for
from skillforge.agents.llm_improver import (
    build_shared_context,
    find_entrypoint,
    parse_code_reply,
    parse_verdict,
)
from skillforge.agents.sandbox import evaluate_source
from skillforge.sim import make_env


def code_reply(code: str, why: str = "Track the cube.") -> str:
    return f"Here is the file.\n```python\n{code}\n```\nRATIONALE: {why}"


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class StubClient:
    """Mimics client.beta.messages.stream(...).get_final_message() closely enough for the improver."""

    def __init__(self, replies, stop_reason="end_turn"):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0)
        cached = len(self.calls) > 1
        usage = SimpleNamespace(
            input_tokens=200, output_tokens=1000,
            cache_creation_input_tokens=0 if cached else 5000, cache_read_input_tokens=5000 if cached else 0,
        )
        content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
        return _Stream(SimpleNamespace(content=content, usage=usage, stop_reason=self.stop_reason, model=kw["model"]))


def test_propose_plans_codes_retries_with_traceback(stub_task):
    client = StubClient(["1. Stop giving up on low-y cubes.", code_reply(BROKEN), code_reply(GOOD, "Removed the give-up branch.")])
    imp = ClaudeImprover(client=client, max_fix_attempts=2)
    cur = src(PARTIAL)
    rep = evaluate_source(stub_task, cur, make_env, range(10)).report
    cand = imp.propose(stub_task, cur, rep)

    assert cand.code.strip() == GOOD.strip() and cand.version == "0.1.1" and cand.parent_version == "0.1.0"
    assert cand.entrypoint == "Skill" and cand.rationale == "Removed the give-up branch."
    assert len(client.calls) == 3
    planner, _coder1, coder2 = client.calls
    for kw in client.calls:
        assert kw["model"] == "claude-opus-5-5"
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
        assert "temperature" not in kw and "budget_tokens" not in str(kw["thinking"])
    assert planner["system"] == coder2["system"]  # one shared, cacheable prefix across roles
    assert "ROLE: planner" in planner["messages"][0]["content"]
    assert rep.summary() in planner["messages"][0]["content"]
    assert "randomization: {" in planner["messages"][0]["content"]
    # the retry carries the sandbox traceback, and history is append-only
    assert [m["role"] for m in coder2["messages"]] == ["user", "assistant", "user"]
    assert "SyntaxError" in coder2["messages"][2]["content"]

    t = imp.usage.totals()
    assert t["calls"] == 3 and t["output_tokens"] == 3000 and t["cache_read_input_tokens"] == 10000
    p = price_for("claude-opus-5-5")
    expect = (600 * p.input + 3000 * p.output + 5000 * p.cache_write + 10000 * p.cache_read) / 1e6
    assert t["usd"] == pytest.approx(expect)


def test_coder_gives_up_after_max_attempts(stub_task):
    client = StubClient(["plan", "no code here", code_reply(BROKEN)])
    imp = ClaudeImprover(client=client, max_fix_attempts=1)
    with pytest.raises(ImproverError, match="no valid skill after 2 attempts"):
        imp.propose(stub_task, src(PARTIAL), evaluate_source(stub_task, src(PARTIAL), make_env, range(3)).report)


def test_budget_cap_refuses_before_calling(stub_task):
    client = StubClient(["plan"])
    imp = ClaudeImprover(client=client, budget_usd=0.01)  # worst case of one planner call is > $0.16
    rep = evaluate_source(stub_task, src(PARTIAL), make_env, range(3)).report
    with pytest.raises(BudgetExceeded):
        imp.propose(stub_task, src(PARTIAL), rep)
    assert client.calls == []


def test_budget_meter_accumulates_and_caps():
    m = UsageMeter(budget_usd=1.0)
    m.record("coder", "claude-opus-5-5", "claude-opus-5-5", SimpleNamespace(input_tokens=0, output_tokens=40_000))
    assert m.usd == pytest.approx(0.8)
    m.check(0.1, "x")
    with pytest.raises(BudgetExceeded):
        m.check(0.3, "x")
    # a fallback to a model with no price row is billed at the priciest tier, never under-reported
    assert price_for("claude-opus-5-5", "claude-mystery").output == 50.0


def test_refusal_is_an_error(stub_task):
    imp = ClaudeImprover(client=StubClient(["nope"], stop_reason="refusal"))
    with pytest.raises(ImproverError, match="refusal"):
        imp.plan(stub_task, src(PARTIAL), evaluate_source(stub_task, src(PARTIAL), make_env, range(2)).report)


def test_missing_key_fails_clearly(monkeypatch):
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(MissingCredentials, match="ANTHROPIC_API_KEY"):
        ClaudeImprover()


def test_full_loop_with_planner_coder_critic(stub_task, tmp_path):
    client = StubClient(["plan", code_reply(GOOD), "Generalises. VERDICT: ACCEPT"])
    imp = ClaudeImprover(client=client)
    res = run_loop(stub_task, src(PARTIAL), imp, make_env, rounds=2, rollouts=10, target=0.9, history_path=tmp_path / "h.json")
    assert res.rounds[1].accepted and res.stopped == "target"
    assert "VERDICT: ACCEPT" in res.rounds[1].critic
    assert res.rounds[1].usage["calls"] == 3 and res.rounds[1].usd_total > 0
    assert "ROLE: critic" in client.calls[2]["messages"][0]["content"]


def test_parsers():
    code, why = parse_code_reply("x\n```python\nA=1\n```\n```python\nB=2\n```\nRATIONALE: because")
    assert code == "B=2\n" and why == "because"
    with pytest.raises(ValueError):
        parse_code_reply("no fence")
    assert find_entrypoint(GOOD, "Nope") == "Skill"
    assert parse_verdict("hmm VERDICT: revert") is False and parse_verdict("ok") is None


def test_shared_context_is_deterministic_and_complete(stub_task):
    a, b = build_shared_context(stub_task), build_shared_context(stub_task)
    assert a == b
    for needle in ("class Observation", "class TaskSpec", "class SkillImprover", "class PickPlace", stub_task.id):
        assert needle in a
