# TokenWise

A developer SDK that wraps an existing LLM client and transparently reduces token
spend while preserving output quality — running inside your own environment.

## Status: Phase 0 — Instrumented Pass-Through

Phase 0 does **no optimization**. It is a drop-in wrapper that:

- returns the provider's **native response object, byte-identical** to an
  un-wrapped call;
- **never breaks your call path** — any internal TokenWise failure is caught and
  the original request still fires (fail-open);
- enforces a **latency budget** on its own overhead;
- records **metadata-only telemetry** (no prompt/response text is ever stored)
  and reports it via `savings()`.

This is the skeleton later phases (semantic cache, context pruning, model
routing, prompt compression) hang off — without reworking the abstractions.

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

`client` here wraps any object exposing
`.chat.completions.create(**kwargs)`, so tests use a fake client with no network.

## Run the tests

```bash
pytest -q
```

The suite encodes the Phase 0 definition of done: transparency, fail-open,
latency budget, telemetry accuracy, policy handling, and estimator fallback.

## Layout

```
tokenwise/
  client.py        # TokenWise wrapper: chat(), savings()
  canonical.py     # provider-agnostic request/response model
  adapters/        # provider adapters (openai first; plural by design)
  policy.py        # presets + override merge (stages disabled in P0)
  pipeline.py      # Stage ABC + fail-open / latency-budget runner (empty in P0)
  telemetry.py     # metadata-only CallRecord + savings aggregation
  tokens.py        # provider-first token counting, estimator fallback
  errors.py        # internal error types
tests/             # Phase 0 definition-of-done suite
```

## What's deliberately easy to add next

- **A provider:** one new file in `adapters/`; `client.py` unchanged.
- **An optimization:** one `Stage` subclass appended to the pipeline; fail-open
  and telemetry already wrap it.
- **A telemetry sink** (control plane): a new `TelemetryStore` backend; the
  schema already anticipates `saved_tokens`, `cache_hit`, `escalated`.
