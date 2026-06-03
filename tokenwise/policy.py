"""Policy scaffold.

Phase 2 ships exact-match cache as the first enabled optimization. Invalid policy
is the one failure raised to the caller, before any provider call -- it is a
developer config mistake, not a runtime fault to swallow.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .errors import ConfigError

_VALID_FAIL_MODES = {"open"}  # "closed" reserved for later


@dataclass
class Policy:
    name: str
    latency_budget_ms: int = 40
    fail_mode: str = "open"
    # Per-stage config. All stages disabled in Phase 0.
    stages: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.latency_budget_ms, int) or self.latency_budget_ms <= 0:
            raise ConfigError(
                f"latency_budget_ms must be a positive int, got {self.latency_budget_ms!r}"
            )
        if self.fail_mode not in _VALID_FAIL_MODES:
            raise ConfigError(
                f"fail_mode {self.fail_mode!r} not supported (Phase 0 supports 'open')"
            )
        cache_cfg = self.stages.get("cache", {})
        ttl = cache_cfg.get("ttl")
        if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
            raise ConfigError(f"cache ttl must be a positive int, got {ttl!r}")


def _preset(name: str) -> Policy:
    base_stages = {
        "cache": {"enabled": True, "ttl": 3600, "namespace": "default"},
        "context_pruning": {"enabled": False},
        "compression": {"enabled": False},
        "routing": {"enabled": False},
        "output_shaping": {"enabled": False},
    }
    return Policy(name=name, latency_budget_ms=40, fail_mode="open",
                 stages=copy.deepcopy(base_stages))


PRESETS = {
    "conservative": _preset("conservative"),
    "balanced": _preset("balanced"),
    "aggressive": _preset("aggressive"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_policy(spec: str | Policy | dict) -> Policy:
    """Resolve a preset name, a Policy, or a dict override-spec into a Policy."""
    if isinstance(spec, Policy):
        spec.validate()
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ConfigError(
                f"unknown policy preset {spec!r}; choose from {sorted(PRESETS)}"
            )
        p = copy.deepcopy(PRESETS[spec])
        p.validate()
        return p
    if isinstance(spec, dict):
        base = copy.deepcopy(PRESETS[spec.get("name", "balanced")]
                             if spec.get("name") in PRESETS else PRESETS["balanced"])
        merged_stages = _deep_merge(base.stages, spec.get("stages", {}))
        p = Policy(
            name=spec.get("name", base.name),
            latency_budget_ms=spec.get("latency_budget_ms", base.latency_budget_ms),
            fail_mode=spec.get("fail_mode", base.fail_mode),
            stages=merged_stages,
        )
        p.validate()
        return p
    raise ConfigError(f"cannot resolve policy from {type(spec).__name__}")


def merge_override(base: Policy, override: str | Policy | dict | None) -> Policy:
    """Merge a per-call override over a default policy."""
    if override is None:
        return base
    if isinstance(override, (str, Policy)):
        return load_policy(override)
    if isinstance(override, dict):
        merged = {
            "name": override.get("name", base.name),
            "latency_budget_ms": override.get("latency_budget_ms", base.latency_budget_ms),
            "fail_mode": override.get("fail_mode", base.fail_mode),
            "stages": _deep_merge(base.stages, override.get("stages", {})),
        }
        return load_policy(merged)
    raise ConfigError(f"cannot merge override from {type(override).__name__}")
