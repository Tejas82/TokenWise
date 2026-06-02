"""Provider-agnostic canonical request/response model.

Every adapter and (future) pipeline stage speaks this format. Several fields are
inert in Phase 0 but present so later phases populate them without a schema
migration: ``CanonicalMessage.metadata['protected']`` (pruning/compression),
``CanonicalRequest.intent_hints`` (router), and ``TokenUsage.source``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list  # str, or content-parts list (multimodal-ready)
    metadata: dict = field(default_factory=dict)  # e.g. {"protected": True}


@dataclass
class CanonicalRequest:
    messages: list[CanonicalMessage]
    tools: list[dict] | None = None
    params: dict = field(default_factory=dict)  # temperature, max_tokens, ...
    intent_hints: dict = field(default_factory=dict)  # task_type?, quality_floor?
    provider_origin: str = "openai"
    # Original provider payload, kept for loss-free reconstruction.
    raw: dict = field(default_factory=dict)


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "provider"  # "provider" | "estimated"


@dataclass
class CanonicalResponse:
    content: str | list
    model: str
    usage: TokenUsage
    finish_reason: str | None = None
    # Original provider response object, returned untouched to the caller.
    raw: Any = None
