"""Phase 1 evaluation foundation.

The eval layer is intentionally local and dependency-free for now. It gives
future optimization stages a place to run in shadow, keep replayable samples,
and score raw-vs-optimized equivalence without changing the served response.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from random import random
from re import sub
from time import time
from typing import Any, Callable, Iterator

from .canonical import CanonicalRequest, CanonicalResponse


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(
            _text(part.get("text", part)) if isinstance(part, dict) else _text(part)
            for part in value
        )
    return str(value)


def _normalize_text(value: Any) -> str:
    return sub(r"\s+", " ", _text(value).strip().lower())


def request_fingerprint(req: CanonicalRequest) -> str:
    """Stable metadata-only request fingerprint for replay lookup."""
    parts = [
        req.provider_origin,
        str(req.params.get("model", "")),
        str(len(req.messages)),
        str(bool(req.tools)),
    ]
    parts.extend(f"{m.role}:{_normalize_text(m.content)}" for m in req.messages)
    return sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass
class ReplaySample:
    ts: float
    request_hash: str
    provider_origin: str
    model_requested: str
    policy_name: str
    metadata: dict = field(default_factory=dict)
    payload: dict | None = None


class ReplayStore:
    """In-memory replay sample store.

    By default only hashes and metadata are stored. Callers can opt into payload
    capture and provide a redactor when they are ready to keep replayable text.
    """

    def __init__(
        self,
        *,
        capture_payloads: bool = False,
        redactor: Callable[[CanonicalRequest], dict] | None = None,
    ) -> None:
        self.capture_payloads = capture_payloads
        self.redactor = redactor
        self._samples: list[ReplaySample] = []

    def capture(self, req: CanonicalRequest, *, policy_name: str) -> ReplaySample:
        payload = None
        if self.capture_payloads:
            payload = self.redactor(req) if self.redactor else _default_redacted_payload(req)
        sample = ReplaySample(
            ts=time(),
            request_hash=request_fingerprint(req),
            provider_origin=req.provider_origin,
            model_requested=str(req.params.get("model", "")),
            policy_name=policy_name,
            metadata={
                "message_count": len(req.messages),
                "has_tools": bool(req.tools),
                "intent_hint_keys": sorted(req.intent_hints),
            },
            payload=payload,
        )
        self._samples.append(sample)
        return sample

    def all(self) -> list[ReplaySample]:
        return list(self._samples)


def _default_redacted_payload(req: CanonicalRequest) -> dict:
    return {
        "messages": [
            {
                "role": msg.role,
                "content_hash": sha256(_text(msg.content).encode("utf-8")).hexdigest(),
                "metadata": dict(msg.metadata),
            }
            for msg in req.messages
        ],
        "tools_hash": sha256(str(req.tools or "").encode("utf-8")).hexdigest(),
        "params": dict(req.params),
    }


@dataclass
class QualityScore:
    method: str
    score: float
    passed: bool
    details: dict = field(default_factory=dict)


class QualityScorer:
    """Cheap-first quality scorer layers for shadow/replay comparisons."""

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.threshold = threshold
        self.embedding_fn = embedding_fn

    def score(
        self,
        raw: CanonicalResponse,
        optimized: CanonicalResponse,
        *,
        validator: Callable[[CanonicalResponse], bool] | None = None,
    ) -> QualityScore:
        if validator is not None:
            ok = bool(validator(optimized))
            return QualityScore("validator", 1.0 if ok else 0.0, ok)

        raw_text = _normalize_text(raw.content)
        opt_text = _normalize_text(optimized.content)
        if raw_text == opt_text:
            return QualityScore("normalized_exact_match", 1.0, True)

        if self.embedding_fn is not None:
            score = _cosine(self.embedding_fn(raw_text), self.embedding_fn(opt_text))
            return QualityScore(
                "embedding_similarity",
                score,
                score >= self.threshold,
                {"threshold": self.threshold},
            )

        score = _token_jaccard(raw_text, opt_text)
        return QualityScore(
            "lexical_similarity",
            score,
            score >= self.threshold,
            {"threshold": self.threshold},
        )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class ShadowRecord:
    ts: float
    request_hash: str
    sample_rate: float
    policy_name: str
    raw_model: str
    optimized_model: str
    raw_tokens: int | None
    optimized_tokens: int | None
    saved_tokens: int
    quality: QualityScore
    error: str | None = None


class ShadowStore:
    def __init__(self) -> None:
        self._records: list[ShadowRecord] = []

    def record(self, rec: ShadowRecord) -> None:
        self._records.append(rec)

    def all(self) -> list[ShadowRecord]:
        return list(self._records)


@dataclass
class ShadowConfig:
    sample_rate: float = 1.0

    def should_sample(self) -> bool:
        return self.sample_rate >= 1.0 or random() < self.sample_rate


class EvaluationManager:
    def __init__(
        self,
        *,
        replay_store: ReplayStore | None = None,
        shadow_store: ShadowStore | None = None,
        scorer: QualityScorer | None = None,
    ) -> None:
        self.replay_store = replay_store or ReplayStore()
        self.shadow_store = shadow_store or ShadowStore()
        self.scorer = scorer or QualityScorer()
        self._shadow_stack: list[ShadowConfig] = []

    @contextmanager
    def shadow(self, *, sample_rate: float = 1.0) -> Iterator[None]:
        if sample_rate < 0 or sample_rate > 1:
            raise ValueError("sample_rate must be between 0 and 1")
        self._shadow_stack.append(ShadowConfig(sample_rate=sample_rate))
        try:
            yield
        finally:
            self._shadow_stack.pop()

    @property
    def active_shadow(self) -> ShadowConfig | None:
        return self._shadow_stack[-1] if self._shadow_stack else None
