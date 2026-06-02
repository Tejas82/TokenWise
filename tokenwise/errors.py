"""Internal exception types.

These represent TokenWise's *own* failures, which the fail-open harness catches
and suppresses (the request proceeds unoptimized). They must never escape to the
caller. The one exception is configuration errors (see ``policy.py``), which are
the developer's mistake and are raised before any provider call fires.
"""

from __future__ import annotations


class TokenWiseError(Exception):
    """Base for internal TokenWise failures (suppressed by fail-open)."""


class StageError(TokenWiseError):
    """A pipeline stage failed; the stage is skipped."""


class StageTimeout(TokenWiseError):
    """A stage exceeded its latency budget; the stage is skipped."""


class ConfigError(TokenWiseError):
    """Invalid configuration. Raised to the caller before any provider call."""
