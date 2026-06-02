"""OpenAI chat-completions adapter.

Wraps a user-supplied OpenAI client instance. Designed to tolerate both object-
style responses (openai>=1.x: resp.choices[0].message.content, resp.usage.*) and
dict-style responses, without importing the openai package -- the wrapped client
is whatever the user passed in. This keeps Phase 0 dependency-light and testable
with a fake client.
"""

from __future__ import annotations

from typing import Any

from .base import ProviderAdapter
from ..canonical import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    TokenUsage,
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or dict key, whichever the object supports."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    # Recognized top-level chat-completions params (everything else passes through).
    _PARAM_KEYS = (
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "n", "seed", "response_format",
        "logprobs", "top_logprobs", "stream", "user",
    )

    def __init__(self, client: Any) -> None:
        self._client = client

    # --- inbound -----------------------------------------------------------
    def to_canonical(self, *args, **kwargs) -> CanonicalRequest:
        # Caller uses kwargs: chat(model=..., messages=[...], ...)
        messages = kwargs.get("messages", [])
        canon_msgs = [
            CanonicalMessage(
                role=_get(m, "role", "user"),
                content=_get(m, "content", ""),
                metadata={},
            )
            for m in messages
        ]
        params = {k: kwargs[k] for k in self._PARAM_KEYS if k in kwargs}
        # model is a param too, kept explicitly for clarity
        if "model" in kwargs:
            params["model"] = kwargs["model"]
        tools = kwargs.get("tools")
        intent_hints = kwargs.get("intent_hints", {}) or {}

        # Preserve the exact original call so reconstruction is loss-free.
        raw = {"args": list(args), "kwargs": dict(kwargs)}
        return CanonicalRequest(
            messages=canon_msgs,
            tools=tools,
            params=params,
            intent_hints=intent_hints,
            provider_origin="openai",
            raw=raw,
        )

    # --- outbound ----------------------------------------------------------
    def from_canonical(self, req: CanonicalRequest) -> dict:
        # Start from the original kwargs so nothing is silently dropped, then
        # overlay any canonical edits a stage may have made.
        kwargs = dict(req.raw.get("kwargs", {}))
        # 'intent_hints' is a TokenWise concept, never sent to the provider.
        kwargs.pop("intent_hints", None)

        # If stages mutated messages, reflect them; in Phase 0 they are identical.
        if req.messages is not None:
            kwargs["messages"] = [
                {"role": m.role, "content": m.content} for m in req.messages
            ]
        if req.tools is not None:
            kwargs["tools"] = req.tools
        return kwargs

    def dispatch(self, payload: dict) -> Any:
        # openai>=1.x: client.chat.completions.create(**payload)
        return self._client.chat.completions.create(**payload)

    def parse_response(self, native_resp: Any) -> CanonicalResponse:
        choices = _get(native_resp, "choices", []) or []
        first = choices[0] if choices else None
        message = _get(first, "message")
        content = _get(message, "content", "")
        finish_reason = _get(first, "finish_reason")
        model = _get(native_resp, "model", "")

        usage_obj = _get(native_resp, "usage")
        if usage_obj is not None:
            usage = TokenUsage(
                prompt_tokens=_get(usage_obj, "prompt_tokens"),
                completion_tokens=_get(usage_obj, "completion_tokens"),
                total_tokens=_get(usage_obj, "total_tokens"),
                source="provider",
            )
        else:
            usage = TokenUsage(source="provider")  # totals None -> estimator kicks in

        return CanonicalResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            raw=native_resp,  # the ORIGINAL object, returned byte-identical
        )
