"""Phase 1 evaluation foundation tests.

Shadow mode serves the raw response, runs the optimized path only for local
measurement, and stores replay/quality records without plaintext by default.
"""

from __future__ import annotations

from conftest import FakeOpenAIClient, FakeResponse, FakeUsage

from tokenwise import QualityScorer, TokenWise
from tokenwise.pipeline import Pipeline, Stage


def test_shadow_mode_returns_served_raw_response_and_records_quality():
    client = FakeOpenAIClient()
    expected = FakeResponse(id="served", usage=FakeUsage(8, 4, 12))
    client.next_response = expected
    tw = TokenWise(client)

    with tw.shadow():
        resp = tw.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert resp is expected
    assert client.call_count == 2
    results = tw.shadow_results()
    assert len(results) == 1
    assert results[0].quality.passed is True
    assert results[0].quality.method == "normalized_exact_match"
    assert results[0].raw_tokens == 12
    assert results[0].optimized_tokens == 12
    assert results[0].saved_tokens == 0


class _RewriteStage(Stage):
    name = "rewrite"

    def applies(self, ctx):
        return True

    def run(self, ctx):
        ctx.request.messages[-1].content = "optimized text"
        return ctx.request


def test_shadow_mode_serves_pre_optimization_request():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    tw._pipeline = Pipeline(stages=[_RewriteStage()])

    with tw.shadow():
        tw.chat(model="m", messages=[{"role": "user", "content": "raw text"}])

    assert client.calls[0]["messages"][-1]["content"] == "raw text"
    assert client.calls[1]["messages"][-1]["content"] == "optimized text"


def test_replay_samples_are_metadata_only_by_default():
    client = FakeOpenAIClient()
    tw = TokenWise(client)

    tw.chat(model="m", messages=[{"role": "user", "content": "secret phrase"}])

    samples = tw.replay_samples()
    assert len(samples) == 1
    assert samples[0].payload is None
    assert "secret phrase" not in str(samples[0].__dict__)
    assert samples[0].metadata["message_count"] == 1


def test_shadow_sample_rate_zero_does_not_run_second_call():
    client = FakeOpenAIClient()
    tw = TokenWise(client)

    with tw.shadow(sample_rate=0):
        tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert client.call_count == 1
    assert tw.shadow_results() == []


def test_quality_scorer_cheap_layers():
    scorer = QualityScorer(threshold=0.3)
    raw = FakeResponse()
    same = FakeResponse()
    different = FakeResponse()
    different.choices[0].message.content = "hello planet"

    raw_resp = TokenWise(FakeOpenAIClient())._adapter.parse_response(raw)
    same_resp = TokenWise(FakeOpenAIClient())._adapter.parse_response(same)
    different_resp = TokenWise(FakeOpenAIClient())._adapter.parse_response(different)

    assert scorer.score(raw_resp, same_resp).method == "normalized_exact_match"
    fuzzy = scorer.score(raw_resp, different_resp)
    assert fuzzy.method == "lexical_similarity"
    assert fuzzy.passed is True


def test_quality_scorer_uses_embedding_layer_when_configured():
    def embed(text: str) -> list[float]:
        return [1.0, 0.0] if "hello" in text else [0.0, 1.0]

    scorer = QualityScorer(threshold=0.9, embedding_fn=embed)
    raw = FakeResponse()
    different = FakeResponse()
    different.choices[0].message.content = "hello planet"

    raw_resp = TokenWise(FakeOpenAIClient())._adapter.parse_response(raw)
    different_resp = TokenWise(FakeOpenAIClient())._adapter.parse_response(different)

    score = scorer.score(raw_resp, different_resp)
    assert score.method == "embedding_similarity"
    assert score.passed is True


def test_invalid_shadow_sample_rate_raises():
    tw = TokenWise(FakeOpenAIClient())
    try:
        with tw.shadow(sample_rate=1.5):
            pass
    except ValueError as exc:
        assert "sample_rate" in str(exc)
    else:
        raise AssertionError("expected ValueError")
