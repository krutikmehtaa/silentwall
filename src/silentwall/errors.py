"""Exception hierarchy for SILENTWALL.

Failures split into two kinds with opposite handling. Data failures degrade: one
unparseable filing out of two thousand should not end a run. Correctness failures
fail loudly, because a split leak or a budget overrun that gets swallowed produces
a number that looks fine and is wrong.

SplitLeakageError and BudgetExceededError are never caught anywhere in this
package. That is enforced by a test.
"""

from __future__ import annotations


class SilentwallError(Exception):
    """Base class for everything this package raises."""


# Data failures. Log, degrade, keep going.


class CorpusFetchError(SilentwallError):
    """A network fetch failed after exhausting retries."""


class ParseIncompleteError(SilentwallError):
    """A filing could not be parsed into a complete record."""


class CacheCorruptError(SilentwallError):
    """A cache record failed to decode or its key did not verify."""


# Resource failures. Retry smaller, then defer.


class BackendOOMError(SilentwallError):
    """The model backend ran out of device memory."""


# Correctness failures. Never caught.


class SplitLeakageError(SilentwallError):
    """An eval-split entity was touched during a dev-split phase.

    This means the reported detectability number would be contaminated by
    training-time information. There is no safe way to continue.
    """


class BudgetExceededError(SilentwallError):
    """The planned generation count exceeds the configured ceiling."""


class MatchingInfeasibleError(SilentwallError):
    """Not enough control entities exist to fill the configured target.

    Shipping a smaller corpus than requested would silently widen every
    confidence interval, so this is fatal rather than a warning.
    """


class ConfigError(SilentwallError):
    """Configuration is missing, malformed, or internally inconsistent."""


#: Errors that must propagate. Referenced by the enforcement test.
NEVER_CAUGHT: tuple[type[SilentwallError], ...] = (
    SplitLeakageError,
    BudgetExceededError,
)
