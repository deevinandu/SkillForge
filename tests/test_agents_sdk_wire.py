"""The real Anthropic SDK builds our request without complaint. A mock transport answers; no network."""

import json

import pytest

from skillforge.agents import ClaudeImprover

anthropic = pytest.importorskip("anthropic")
try:
    import httpx2 as httpx
except ImportError:  # older SDKs
    import httpx


def test_sdk_serializes_the_request(stub_task):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["headers"] = dict(req.headers)
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "mock"}})

    client = anthropic.Anthropic(
        api_key="sk-mock-not-real", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler)),
    )
    imp = ClaudeImprover(client=client)
    with pytest.raises(anthropic.APIStatusError):
        imp._call("planner", stub_task, [{"role": "user", "content": "hi"}], 100, "medium")

    b = seen["body"]
    assert b["model"] == "claude-opus-5-5" and b["stream"] is True
    assert b["thinking"] == {"type": "adaptive"} and b["output_config"] == {"effort": "medium"}
    assert b["fallbacks"] == "default"
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "server-side-fallback-2026-07-01" in seen["headers"].get("anthropic-beta", "")
    assert imp.usage.calls == []  # nothing billed for a failed call
