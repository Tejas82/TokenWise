"""Pipeline runner with fail-open execution and per-stage latency budgets.

Phase 0 registers zero stages, but the harness is fully exercised: it wraps the
(empty) stage list, enforces a global overhead budget, and proves that any
internal failure is caught and skipped while the request proceeds. Later phases
append Stage subclasses; the safety net already wraps them.

Note on timeouts: Phase 0 uses a cooperative deadline (checked after each stage),
not signal/thread preemption. A single stage that hangs internally is a stage-
implementation bug; Phase 0 stages are synchronous and fast. Hard preemption is a
hardening-phase concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from .canonical import CanonicalRequest
from .errors import StageError, StageTimeout
from .policy import Policy
from .telemetry import _StageRecorder


@dataclass
class PipelineContext:
    request: CanonicalRequest
    policy: Policy
    recorder: _StageRecorder
    budget_remaining_ms: float


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def applies(self, ctx: PipelineContext) -> bool:
        """Whether this stage is enabled for the current policy/request."""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> CanonicalRequest:
        """Transform and return the request. Must be side-effect free on failure."""


class Pipeline:
    def __init__(self, stages: list[Stage] | None = None) -> None:
        self.stages = stages or []

    def run(self, ctx: PipelineContext) -> CanonicalRequest:
        """Run each enabled stage under fail-open + budget. Returns the
        (possibly unmodified) request. Never raises for stage failures."""
        for stage in self.stages:
            if ctx.budget_remaining_ms <= 0:
                ctx.recorder.note_stage_skip(stage.name, "budget exhausted")
                continue
            try:
                if not _safe_applies(stage, ctx):
                    continue
            except Exception as e:  # noqa: BLE001 - fail-open by design
                ctx.recorder.note_stage_skip(stage.name, f"applies() error: {e}")
                continue
            ctx.request = _run_protected(stage, ctx)
        return ctx.request


def _safe_applies(stage: Stage, ctx: PipelineContext) -> bool:
    return bool(stage.applies(ctx))


def _run_protected(stage: Stage, ctx: PipelineContext) -> CanonicalRequest:
    start = monotonic()
    prior = ctx.request
    try:
        result = stage.run(ctx)
        elapsed_ms = (monotonic() - start) * 1000.0
        ctx.recorder.note_stage_latency(stage.name, elapsed_ms)
        ctx.budget_remaining_ms -= elapsed_ms
        if result is None:
            ctx.recorder.note_stage_skip(stage.name, "returned None")
            return prior
        return result
    except (StageError, StageTimeout) as e:
        elapsed_ms = (monotonic() - start) * 1000.0
        ctx.recorder.note_stage_latency(stage.name, elapsed_ms)
        ctx.budget_remaining_ms -= elapsed_ms
        ctx.recorder.note_stage_skip(stage.name, str(e))
        return prior
    except Exception as e:  # noqa: BLE001 - fail-open: swallow ALL stage faults
        elapsed_ms = (monotonic() - start) * 1000.0
        ctx.recorder.note_stage_latency(stage.name, elapsed_ms)
        ctx.budget_remaining_ms -= elapsed_ms
        ctx.recorder.note_stage_skip(stage.name, f"unhandled: {e}")
        return prior


def run_callable_protected(
    fn: Callable[[], object], recorder: _StageRecorder, name: str
) -> object | None:
    """Fail-open wrapper for non-stage internal steps (e.g. normalization)."""
    start = monotonic()
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        recorder.note_stage_skip(name, f"unhandled: {e}")
        return None
    finally:
        recorder.note_stage_latency(name, (monotonic() - start) * 1000.0)
