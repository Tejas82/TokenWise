"""Phase 2 exact-match cache tests."""

from __future__ import annotations

from conftest import FakeOpenAIClient, FakeResponse, FakeUsage

from tokenwise import TokenWise, exact_cache_key, load_policy


def test_exact_cache_hit_skips_provider_and_records_savings():
    client = FakeOpenAIClient()
    client.next_response = FakeResponse(usage=FakeUsage(7, 3, 10))
    tw = TokenWise(client)

    first = tw.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    second = tw.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert client.call_count == 1
    assert second.choices[0].message.content == first.choices[0].message.content

    first_rec, second_rec = tw.telemetry.all()
    assert first_rec.cache_hit is False
    assert second_rec.cache_hit is True
    assert second_rec.usage.source == "cache"
    assert second_rec.usage.total_tokens == 0
    assert second_rec.baseline_tokens == 10
    assert second_rec.saved_tokens == 10

    report = tw.savings()
    assert report.calls == 2
    assert report.total_tokens == 10
    assert report.total_saved_tokens == 10
    assert report.cache_hits == 1
    assert report.cache_hit_rate == 50.0


def test_cache_can_be_disabled_by_policy_override():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    policy = {"stages": {"cache": {"enabled": False}}}

    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], policy=policy)
    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], policy=policy)

    assert client.call_count == 2
    assert all(not rec.cache_hit for rec in tw.telemetry.all())


def test_different_params_do_not_hit_exact_cache():
    client = FakeOpenAIClient()
    tw = TokenWise(client)

    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], temperature=0.1)
    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], temperature=0.7)

    assert client.call_count == 2
    assert tw.savings().cache_hits == 0


def test_streaming_requests_are_not_cached():
    client = FakeOpenAIClient()
    tw = TokenWise(client)

    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)

    assert client.call_count == 2
    assert tw.cache.size() == 0


def test_exact_cache_key_is_stable_for_equivalent_requests():
    client = FakeOpenAIClient()
    tw = TokenWise(client)
    req1 = tw._adapter.to_canonical(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    req2 = tw._adapter.to_canonical(
        messages=[{"content": "hi", "role": "user"}],
        model="m",
    )
    policy = load_policy("balanced")

    assert exact_cache_key(req1, policy_name=policy.name) == exact_cache_key(
        req2,
        policy_name=policy.name,
    )


def test_shadow_mode_uses_cache_for_optimized_path_after_warmup():
    client = FakeOpenAIClient()
    client.next_response = FakeResponse(usage=FakeUsage(8, 4, 12))
    tw = TokenWise(client)

    tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    with tw.shadow():
        tw.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert client.call_count == 2
    shadow = tw.shadow_results()[-1]
    assert shadow.raw_tokens == 12
    assert shadow.optimized_tokens == 0
    assert shadow.saved_tokens == 12
    assert shadow.quality.passed is True
