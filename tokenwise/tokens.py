"""Token counting: trust provider-reported usage, fall back to a local estimate.

The estimator is intentionally simple and dependency-free in Phase 0 (a rough
chars/4 heuristic). It exists so future providers and streaming responses that
omit inline usage still produce a (clearly flagged) number. It can be swapped for
a real tokenizer later without changing callers.
"""

from __future__ import annotations

from .canonical import CanonicalRequest, CanonicalResponse, TokenUsage

_CHARS_PER_TOKEN = 4  # crude but stable heuristic for the fallback path


def _text_of(content: str | list) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(str(item.get("text", "")))
    return " ".join(parts)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_usage(req: CanonicalRequest, resp: CanonicalResponse) -> TokenUsage:
    prompt_text = " ".join(_text_of(m.content) for m in req.messages)
    completion_text = _text_of(resp.content)
    p = estimate_tokens(prompt_text)
    c = estimate_tokens(completion_text)
    return TokenUsage(
        prompt_tokens=p,
        completion_tokens=c,
        total_tokens=p + c,
        source="estimated",
    )


def usage_for(req: CanonicalRequest, resp: CanonicalResponse) -> TokenUsage:
    """Prefer provider usage; estimate only when it is absent."""
    if resp.usage is not None and resp.usage.total_tokens is not None:
        return resp.usage
    return estimate_usage(req, resp)
