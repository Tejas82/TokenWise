"""Phase 0 definition-of-done tests (six core checks + supporting cases).

1. Transparency      - wrapped response is byte-identical to native
2. Fail-open         - internal failures never break the provider call
3. Latency budget    - over-budget stage is skipped, request proceeds
4. Telemetry accuracy- savings() totals match provider usage; saved==0
5. Policy            - presets load, overrides merge, bad config raises early
6. Estimator fallback- missing provider usage -> estimated, flagged
"""

from __future__ import annotations

import pytest

from conftest import FakeOpenAIClient, FakeResponse, FakeUsage

from tokenwise import TokenWise, load_policy, PRESETS
from tokenwise.errors import ConfigError
from tokenwise.pipeline import Pipeline, PipelineContext, Stage
from tokenwise.canonical import CanonicalRequest


# --- 1. Transparency -------------------------------------------------------
def test_response_is_byte_identical_native_object():
    client = FakeOpenAIClient()
    expected = FakeResponse(id="resp_unique", model="gpt-4o-mini")
    client.next_response = expected

    tw = TokenWise(client)
    resp = tw.chat(model="gpt-4o-mini",
                   messages=[{"role": "user", "content": "hi"}])

    assert resp is expected  # same object, not a copy
    assert resp.choices[0].message.content == "hello world"
    assert resp.choices[0].finish_reason == "stop"


def test_transparency_with_tools_and_history():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    tw.chat(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        tools=[{"type": "function", "function": {"name": "f"}}],
        temperature=0.2,
    )
    sent = client.last_call
    assert sent["model"] == "gpt-4o-mini"
    assert len(sent["messages"]) == 4
    assert sent["tools"][0]["function"]["name"] == "f"
    assert sent["temperature"] == 0.2
    # intent_hints must never be forwarded to the provider
    assert "intent_hints" not in sent


def test_intent_hints_stripped_from_provider_call():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    tw.chat(model="m", messages=[{"role": "user", "content": "x"}],
            intent_hints={"task_type": "extraction"})
    assert "intent_hints" not in client.last_call


# --- 2. Fail-open ----------------------------------------------------------
class _BoomStage(Stage):
    name = "boom"

    def applies(self, ctx):
        return True

    def run(self, ctx):
        raise RuntimeError("stage blew up")


def test_failing_stage_does_not_break_call():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    tw._pipeline = Pipeline(stages=[_BoomStage()])  # inject a failing stage

    resp = tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert resp is not None
    assert client.call_count == 1  # provider still called
    rec = tw.telemetry.all()[-1]
    assert any(s["stage"] == "boom" for s in rec.stage_skips)


def test_normalization_failure_falls_back_to_raw_dispatch(monkeypatch):
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    # Force to_canonical to fail.
    monkeypatch.setattr(
        tw._adapter, "to_canonical",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("normalize fail")),
    )
    resp = tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert resp is not None
    assert client.call_count == 1
    # The raw kwargs were dispatched verbatim.
    assert client.last_call["model"] == "m"


def test_provider_error_is_NOT_suppressed():
    client = FakeOpenAIClient()
    client.raise_on_call = True
    tw = TokenWise(client)
    with pytest.raises(RuntimeError, match="provider exploded"):
        tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])


# --- 3. Latency budget -----------------------------------------------------
class _SlowStage(Stage):
    name = "slow"

    def applies(self, ctx):
        return True

    def run(self, ctx):
        import time as _t
        _t.sleep(0.02)  # 20ms
        return ctx.request


class _AfterStage(Stage):
    name = "after"

    def __init__(self):
        self.ran = False

    def applies(self, ctx):
        return True

    def run(self, ctx):
        self.ran = True
        return ctx.request


def test_budget_exhaustion_skips_later_stages():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    after = _AfterStage()
    tw._pipeline = Pipeline(stages=[_SlowStage(), after])
    # Tiny budget so the slow stage exhausts it.
    tw._policy = load_policy({"name": "balanced", "latency_budget_ms": 5})

    resp = tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert resp is not None
    rec = tw.telemetry.all()[-1]
    assert any(s["stage"] == "after" and "budget" in s["reason"]
               for s in rec.stage_skips)
    assert after.ran is False


# --- 4. Telemetry accuracy -------------------------------------------------
def test_savings_totals_match_provider_usage():
    client = FakeOpenAIClient()
    client.next_response = FakeResponse(usage=FakeUsage(7, 3, 10))
    tw = TokenWise(client, policy={"stages": {"cache": {"enabled": False}}})
    for _ in range(3):
        tw.chat(model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}])
    report = tw.savings()
    assert report.calls == 3
    assert report.total_tokens == 30  # 10 * 3
    assert report.total_saved_tokens == 0
    assert report.by_model["gpt-4o-mini"]["calls"] == 3


def test_no_payload_stored_in_telemetry():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    tw.chat(model="m", messages=[{"role": "user", "content": "secret data"}])
    rec = tw.telemetry.all()[-1]
    # Serialize the record's dict and assert the payload text is absent.
    blob = str(rec.__dict__)
    assert "secret data" not in blob
    assert "hello world" not in blob


# --- 5. Policy -------------------------------------------------------------
def test_all_presets_load():
    for name in ("conservative", "balanced", "aggressive"):
        p = load_policy(name)
        assert p.name == name
        assert p.stages["cache"]["enabled"] is True
        assert p.stages["cache"]["ttl"] == 3600
        assert all(
            not cfg["enabled"]
            for stage, cfg in p.stages.items()
            if stage != "cache"
        )


def test_per_call_override_merges():
    base = load_policy("balanced")
    merged = load_policy({
        "name": "balanced",
        "latency_budget_ms": 100,
        "stages": {"cache": {"enabled": True, "ttl": 60}},
    })
    assert merged.latency_budget_ms == 100
    assert merged.stages["cache"]["enabled"] is True
    assert merged.stages["cache"]["ttl"] == 60
    # untouched stage remains disabled
    assert merged.stages["routing"]["enabled"] is False


def test_invalid_policy_raises_before_call():
    with pytest.raises(ConfigError):
        load_policy("does_not_exist")
    with pytest.raises(ConfigError):
        load_policy({"name": "balanced", "latency_budget_ms": -1})


def test_bad_override_raises_and_no_provider_call():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    with pytest.raises(ConfigError):
        tw.chat(model="m",
                messages=[{"role": "user", "content": "hi"}],
                policy={"name": "balanced", "latency_budget_ms": 0})
    assert client.call_count == 0  # raised before dispatch


# --- 6. Estimator fallback -------------------------------------------------
def test_estimator_used_when_usage_absent():
    client = FakeOpenAIClient()
    resp = FakeResponse()
    resp.usage = None  # provider returned no usage
    client.next_response = resp
    tw = TokenWise(client)
    tw.chat(model="m",
            messages=[{"role": "user", "content": "a longer message here"}])
    rec = tw.telemetry.all()[-1]
    assert rec.usage.source == "estimated"
    assert rec.usage.total_tokens is not None
    assert rec.usage.total_tokens > 0
