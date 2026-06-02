"""Provider adapter contract.

Adapters translate between a provider's native call/response and the canonical
model. The directory is plural from day one so a second provider is a new file,
not a refactor. In Phase 0 with no stages active, the round trip
to_canonical -> from_canonical -> dispatch -> parse_response must reproduce the
call the user would have made directly, and the caller receives the provider's
ORIGINAL native response object (byte-identical).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..canonical import CanonicalRequest, CanonicalResponse


class ProviderAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def to_canonical(self, *args, **kwargs) -> CanonicalRequest:
        ...

    @abstractmethod
    def from_canonical(self, req: CanonicalRequest) -> dict:
        """Rebuild the provider-native payload (kwargs) for dispatch."""

    @abstractmethod
    def dispatch(self, payload: dict) -> Any:
        """Call the underlying provider client; return its native response."""

    @abstractmethod
    def parse_response(self, native_resp: Any) -> CanonicalResponse:
        ...
