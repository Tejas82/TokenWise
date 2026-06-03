"""Phase 2 exact-match cache.

The first optimization is intentionally conservative: only byte-stable canonical
request matches are served from cache. Semantic lookup comes later behind this
same store shape.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from time import time
from typing import Any

from .canonical import CanonicalRequest, TokenUsage


def exact_cache_key(
    req: CanonicalRequest,
    *,
    namespace: str = "default",
    policy_name: str = "",
) -> str:
    payload = {
        "namespace": namespace,
        "policy": policy_name,
        "provider_origin": req.provider_origin,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "metadata": msg.metadata,
            }
            for msg in req.messages
        ],
        "tools": req.tools,
        "params": req.params,
        "intent_hints": req.intent_hints,
    }
    blob = dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    key: str
    response: Any
    usage: TokenUsage
    created_at: float
    expires_at: float | None = None

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time()) >= self.expires_at


class ExactCacheStore:
    """In-memory exact-match cache for Phase 2."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expired():
            self._entries.pop(key, None)
            return None
        return entry

    def set(
        self,
        key: str,
        response: Any,
        usage: TokenUsage,
        *,
        ttl: int | None = None,
    ) -> CacheEntry:
        now = time()
        entry = CacheEntry(
            key=key,
            response=_safe_deepcopy(response),
            usage=deepcopy(usage),
            created_at=now,
            expires_at=now + ttl if ttl else None,
        )
        self._entries[key] = entry
        return entry

    def clear(self) -> None:
        self._entries.clear()

    def size(self) -> int:
        return len(self._entries)


def cached_response(entry: CacheEntry) -> Any:
    return _safe_deepcopy(entry.response)


def _safe_deepcopy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:  # noqa: BLE001 - native provider objects can be unusual
        return value
