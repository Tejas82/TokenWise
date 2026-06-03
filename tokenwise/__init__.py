"""TokenWise SDK -- instrumented pass-through plus local eval foundation."""

from .client import TokenWise
from .cache import CacheEntry, ExactCacheStore, exact_cache_key
from .eval import (
    EvaluationManager,
    QualityScore,
    QualityScorer,
    ReplaySample,
    ReplayStore,
    ShadowRecord,
    ShadowStore,
)
from .policy import Policy, load_policy, PRESETS
from .telemetry import TelemetryStore, SavingsReport, CallRecord
from .canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalMessage,
    TokenUsage,
)

__version__ = "0.0.1"

__all__ = [
    "TokenWise",
    "CacheEntry",
    "ExactCacheStore",
    "exact_cache_key",
    "EvaluationManager",
    "QualityScore",
    "QualityScorer",
    "ReplaySample",
    "ReplayStore",
    "ShadowRecord",
    "ShadowStore",
    "Policy",
    "load_policy",
    "PRESETS",
    "TelemetryStore",
    "SavingsReport",
    "CallRecord",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalMessage",
    "TokenUsage",
]
