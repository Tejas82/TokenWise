# TokenWise

A developer SDK that wraps an existing LLM client and transparently reduces token
spend while preserving output quality — running inside your own environment.

## Status: Phase 2 — Exact-Match Cache

TokenWise now has the Phase 0/1 safety foundation plus the first real
optimization: an in-process exact-match cache. It:

- returns the provider's **native response object, byte-identical** to an
  un-wrapped call on cache misses;
- **never breaks your call path** — any internal TokenWise failure is caught and
  the original request still fires (fail-open);
- enforces a **latency budget** on its own overhead;
- records **metadata-only telemetry** (no prompt/response text is ever stored)
- supports **shadow mode** for raw-vs-optimized evaluation;
- serves repeated identical requests from a local **exact-match cache** and
  reports avoided provider tokens via `savings()`.

This is still intentionally conservative. Semantic cache, context pruning, model
routing, and prompt compression come later, behind the same fail-open/eval
machinery.

## Install (local, editable)

```bash
cd "TokenWise"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from openai import OpenAI
from tokenwise import TokenWise

client = TokenWise(OpenAI(), policy="balanced")

resp = client.chat(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# resp is the native OpenAI response — use it exactly as before:
print(resp.choices[0].message.content)

print(client.savings())   # token totals + overhead, metadata only
```

Repeated identical calls are served from the local exact-match cache by default:

```python
client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}])
client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}])

report = client.savings()
print(report.cache_hits)
print(report.total_saved_tokens)
```

Disable cache per client or per call when you need pure pass-through behavior:

```python
client = TokenWise(OpenAI(), policy={"stages": {"cache": {"enabled": False}}})

resp = client.chat(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    policy={"stages": {"cache": {"enabled": False}}},
)
```

Use shadow mode to evaluate optimizations without serving optimized output:

```python
with client.shadow():
    client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}])

for result in client.shadow_results():
    print(result.quality.passed, result.saved_tokens)
```

`client` here wraps any object exposing
`.chat.completions.create(**kwargs)`, so tests use a fake client with no network.

## Run the tests

```bash
.venv/bin/pytest
```

The suite covers Phase 0 pass-through safety, Phase 1 evaluation, and Phase 2
exact-cache behavior.

## Layout

```
tokenwise/
  client.py        # TokenWise wrapper: chat(), savings()
  cache.py         # exact-match cache store + keying
  canonical.py     # provider-agnostic request/response model
  eval.py          # shadow mode, replay samples, quality scoring
  adapters/        # provider adapters (openai first; plural by design)
  policy.py        # presets + override merge
  pipeline.py      # Stage ABC + fail-open / latency-budget runner
  telemetry.py     # metadata-only CallRecord + savings aggregation
  tokens.py        # provider-first token counting, estimator fallback
  errors.py        # internal error types
tests/             # Phase 0-2 behavior suite
```

## What's deliberately easy to add next

- **A provider:** one new file in `adapters/`; `client.py` unchanged.
- **Semantic cache:** extend `cache.py` with embedding lookup after exact-match
  misses.
- **Context pruning:** add a request-transforming stage before dispatch.
- **A telemetry sink** (control plane): a new `TelemetryStore` backend; the
  schema already anticipates `saved_tokens`, `cache_hit`, `escalated`.
