"""The TokenWise drop-in wrapper.

Phase 0 contract: chat() returns the provider's NATIVE response object (byte-
identical to an un-wrapped call). The canonical form is internal plumbing. The
pipeline is empty, so the only observable effect is telemetry. Any internal
failure is swallowed (fail-open) and the provider call still fires; only config
errors surface, and they do so before any call.
"""

from __future__ import annotations

from time import monotonic, time
from typing import Any

from .adapters import OpenAIAdapter
from .pipeline import Pipeline, PipelineContext, run_callable_protected
from .policy import Policy, load_policy, merge_override
from .telemetry import CallRecord, SavingsReport, TelemetryStore, _StageRecorder
from .tokens import usage_for


class TokenWise:
    def __init__(
        self,
        provider_client: Any,
        policy: str | Policy | dict = "balanced",
        telemetry: TelemetryStore | None = None,
    ) -> None:
        self._adapter = OpenAIAdapter(provider_client)
        self._policy: Policy = load_policy(policy)  # config errors raise here
        self._pipeline = Pipeline(stages=[])  # empty in Phase 0
        self._telemetry = telemetry or TelemetryStore()

    @property
    def telemetry(self) -> TelemetryStore:
        return self._telemetry

    def chat(
        self,
        *args,
        policy: str | Policy | dict | None = None,
        **kwargs,
    ) -> Any:
        # Resolve effective policy first; a bad per-call override should raise
        # before we touch the provider (it's a developer config mistake).
        effective = merge_override(self._policy, policy)

        recorder = _StageRecorder()
        pipeline_start = monotonic()

        # 1. Normalize (fail-open: on failure we fall back to a raw dispatch).
        req = run_callable_protected(
            lambda: self._adapter.to_canonical(*args, **kwargs),
            recorder,
            "normalize",
        )

        if req is not None:
            # 2-3. Run the (empty) pipeline under fail-open + latency budget.
            ctx = PipelineContext(
                request=req,
                policy=effective,
                recorder=recorder,
                budget_remaining_ms=float(effective.latency_budget_ms),
            )
            req = self._pipeline.run(ctx)

        overhead_ms = (monotonic() - pipeline_start) * 1000.0

        # 4. Dispatch. The provider call is NOT fail-open-suppressed: provider
        #    errors are the caller's to handle and surface normally.
        if req is not None:
            payload = self._adapter.from_canonical(req)
        else:
            # Normalization failed -> dispatch the user's original call verbatim.
            payload = dict(kwargs)
        native_resp = self._adapter.dispatch(payload)

        # 5. Parse for telemetry only; the caller gets the native object back.
        try:
            canon_resp = self._adapter.parse_response(native_resp)
            usage = usage_for(req, canon_resp) if req is not None else canon_resp.usage
            total = usage.total_tokens
            self._telemetry.record(
                CallRecord(
                    ts=time(),
                    model_requested=str(kwargs.get("model", "")),
                    model_used=canon_resp.model or str(kwargs.get("model", "")),
                    usage=usage,
                    baseline_tokens=total,  # == actual in Phase 0
                    saved_tokens=0,
                    overhead_ms=round(overhead_ms, 4),
                    stage_latencies=recorder.stage_latencies,
                    stage_skips=recorder.stage_skips,
                    cache_hit=False,
                    escalated=False,
                    policy_name=effective.name,
                )
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the call
            pass

        # 6. Return the original native response, untouched.
        return native_resp

    def savings(self, window: str = "all") -> SavingsReport:
        return self._telemetry.savings(window)
