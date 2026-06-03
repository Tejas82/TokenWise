"""The TokenWise drop-in wrapper.

Phase 0 contract: chat() returns the provider's NATIVE response object (byte-
identical to an un-wrapped call). The canonical form is internal plumbing. The
pipeline is empty, so the only observable effect is telemetry. Any internal
failure is swallowed (fail-open) and the provider call still fires; only config
errors surface, and they do so before any call.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import AbstractContextManager
from time import monotonic, time
from typing import Any

from .adapters import OpenAIAdapter
from .eval import EvaluationManager, QualityScore, ShadowRecord
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
        evaluation: EvaluationManager | None = None,
    ) -> None:
        self._adapter = OpenAIAdapter(provider_client)
        self._policy: Policy = load_policy(policy)  # config errors raise here
        self._pipeline = Pipeline(stages=[])  # empty in Phase 0
        self._telemetry = telemetry or TelemetryStore()
        self._evaluation = evaluation or EvaluationManager()

    @property
    def telemetry(self) -> TelemetryStore:
        return self._telemetry

    @property
    def evaluation(self) -> EvaluationManager:
        return self._evaluation

    def shadow(self, *, sample_rate: float = 1.0) -> AbstractContextManager[None]:
        """Run optimized calls beside raw calls and discard optimized output.

        The caller always receives the raw provider response while Phase 1 records
        local quality/savings evidence for future optimization stages.
        """
        return self._evaluation.shadow(sample_rate=sample_rate)

    def replay_samples(self):
        return self._evaluation.replay_store.all()

    def shadow_results(self):
        return self._evaluation.shadow_store.all()

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

        raw_req = deepcopy(req) if req is not None else None
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
        shadow_config = self._evaluation.active_shadow
        if (
            req is not None
            and raw_req is not None
            and shadow_config is not None
            and shadow_config.should_sample()
        ):
            native_resp = self._dispatch_shadowed(
                raw_req,
                req,
                effective_policy_name=effective.name,
                sample_rate=shadow_config.sample_rate,
            )
        elif req is not None:
            payload = self._adapter.from_canonical(req)
            self._evaluation.replay_store.capture(req, policy_name=effective.name)
            native_resp = self._adapter.dispatch(payload)
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

    def _dispatch_shadowed(
        self,
        served_req,
        optimized_req,
        *,
        effective_policy_name: str,
        sample_rate: float,
    ) -> Any:
        # The raw request is served. The optimized path is discarded and used
        # only for local evaluation records.
        self._evaluation.replay_store.capture(served_req, policy_name=effective_policy_name)
        sample = self._evaluation.replay_store.all()[-1]

        native_resp = self._adapter.dispatch(self._adapter.from_canonical(served_req))
        try:
            raw_resp = self._adapter.parse_response(native_resp)
            optimized_native = self._adapter.dispatch(
                self._adapter.from_canonical(optimized_req)
            )
            optimized_resp = self._adapter.parse_response(optimized_native)
            raw_usage = usage_for(served_req, raw_resp)
            optimized_usage = usage_for(optimized_req, optimized_resp)
            quality = self._evaluation.scorer.score(raw_resp, optimized_resp)
            raw_tokens = raw_usage.total_tokens
            optimized_tokens = optimized_usage.total_tokens
            saved = (
                max(raw_tokens - optimized_tokens, 0)
                if raw_tokens is not None and optimized_tokens is not None
                else 0
            )
            self._evaluation.shadow_store.record(
                ShadowRecord(
                    ts=time(),
                    request_hash=sample.request_hash,
                    sample_rate=sample_rate,
                    policy_name=effective_policy_name,
                    raw_model=raw_resp.model,
                    optimized_model=optimized_resp.model,
                    raw_tokens=raw_tokens,
                    optimized_tokens=optimized_tokens,
                    saved_tokens=saved,
                    quality=quality,
                )
            )
        except Exception as exc:  # noqa: BLE001 - shadow must never affect prod
            raw_model = ""
            raw_tokens = None
            quality = QualityScore("shadow_error", 0.0, False)
            try:
                raw_resp = self._adapter.parse_response(native_resp)
                raw_model = raw_resp.model
                raw_tokens = usage_for(served_req, raw_resp).total_tokens
                quality = self._evaluation.scorer.score(raw_resp, raw_resp)
            except Exception:  # noqa: BLE001
                pass
            self._evaluation.shadow_store.record(
                ShadowRecord(
                    ts=time(),
                    request_hash=sample.request_hash,
                    sample_rate=sample_rate,
                    policy_name=effective_policy_name,
                    raw_model=raw_model,
                    optimized_model="",
                    raw_tokens=raw_tokens,
                    optimized_tokens=None,
                    saved_tokens=0,
                    quality=quality,
                    error=str(exc),
                )
            )
        return native_resp
