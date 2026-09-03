"""SILENTWALL: audit the detectability of information barriers in LLM agents.

The headline question this package answers is not "did the agent forget" but
"can an observer tell that anything was forgotten". A barrier that visibly
changes the agent's behaviour on restricted entities leaks the restricted list
itself, which in a financial compliance setting is exactly the thing being
protected.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import SilentwallConfig, load_config
from .types import (
    AuditResult,
    Corpus,
    DealRecord,
    DetectabilityResult,
    Entity,
    Generation,
    LeakResult,
    Probe,
    ProbeFamily,
)

__all__ = [
    "__version__",
    "SilentwallConfig",
    "load_config",
    "AuditResult",
    "Corpus",
    "DealRecord",
    "DetectabilityResult",
    "Entity",
    "Generation",
    "LeakResult",
    "Probe",
    "ProbeFamily",
]
