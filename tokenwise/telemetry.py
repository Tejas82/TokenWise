"""Metadata-only telemetry.

No prompt or completion text is ever stored -- this is the governance precedent
that the metadata path and the payload path are separate. The CallRecord schema
includes fields that are always zero/false in Phase 0 (saved_tokens, cache_hit,
escalated, baseline_tokens) so later phases populate existing fields rather than
forcing a migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from .canonical import TokenUsage


@dataclass
class CallRecord:
    ts: float
    model_requested: str
    model_used: str  # == requested in Phase 0
    usage: TokenUsage
    baseline_tokens: int | None = None  # == actual in Phase 0
    saved_tokens: int = 0  # 0 in Phase 0
    overhead_ms: float = 0.0
    stage_latencies: dict = field(default_factory=dict)
    stage_skips: list = field(default_factory=list)
    cache_hit: bool = False  # always False in Phase 0
    escalated: bool = False  # always False in Phase 0
    policy_name: str = ""
    # NB: no prompt/response payload fields, ever.


@dataclass
class SavingsReport:
    calls: int
    total_tokens: int
    total_saved_tokens: int
    by_model: dict
    avg_overhead_ms: float
    estimated_usage_calls: int
    cache_hits: int = 0
    cache_hit_rate: float = 0.0

    def __str__(self) -> str:
        baseline = self.total_tokens + self.total_saved_tokens
        pct = (100.0 * self.total_saved_tokens / baseline) if baseline else 0.0
        return (
            f"TokenWise savings: {self.calls} calls, "
            f"{self.total_tokens} tokens, "
            f"{self.total_saved_tokens} saved ({pct:.1f}%), "
            f"{self.cache_hits} cache hits ({self.cache_hit_rate:.1f}%), "
            f"avg overhead {self.avg_overhead_ms:.2f}ms"
        )


class _StageRecorder:
    """Per-call scratch object stages write to (latencies, skips)."""

    def __init__(self) -> None:
        self.stage_latencies: dict = {}
        self.stage_skips: list = []

    def note_stage_latency(self, name: str, ms: float) -> None:
        self.stage_latencies[name] = round(ms, 4)

    def note_stage_skip(self, name: str, reason: str) -> None:
        self.stage_skips.append({"stage": name, "reason": reason})


class TelemetryStore:
    """In-memory store for Phase 0. Later phases add control-plane sinks behind
    this same interface."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        self._records.append(rec)

    def all(self) -> list[CallRecord]:
        return list(self._records)

    def savings(self, window: str = "all") -> SavingsReport:
        recs = self._records
        if window != "all":
            cutoff = _window_cutoff(window)
            recs = [r for r in recs if r.ts >= cutoff]

        total_tokens = sum((r.usage.total_tokens or 0) for r in recs)
        total_saved = sum(r.saved_tokens for r in recs)
        by_model: dict = {}
        for r in recs:
            m = by_model.setdefault(r.model_used, {"calls": 0, "tokens": 0})
            m["calls"] += 1
            m["tokens"] += r.usage.total_tokens or 0
        avg_overhead = (sum(r.overhead_ms for r in recs) / len(recs)) if recs else 0.0
        est_calls = sum(1 for r in recs if r.usage.source == "estimated")
        cache_hits = sum(1 for r in recs if r.cache_hit)

        return SavingsReport(
            calls=len(recs),
            total_tokens=total_tokens,
            total_saved_tokens=total_saved,
            by_model=by_model,
            avg_overhead_ms=round(avg_overhead, 4),
            estimated_usage_calls=est_calls,
            cache_hits=cache_hits,
            cache_hit_rate=round(100.0 * cache_hits / len(recs), 4) if recs else 0.0,
        )


def _window_cutoff(window: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        n, u = int(window[:-1]), window[-1]
        return time() - n * units[u]
    except (ValueError, KeyError):
        return 0.0
