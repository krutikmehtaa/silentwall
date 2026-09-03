"""Containment methods.

Importing this package registers every built-in method, so the registry is populated
by the time the harness asks for one.
"""

from __future__ import annotations

from .base import BaseContainment, ContainmentMethod, EntityContext
from .baselines import (
    CleanReference,
    LoraGradientAscent,
    NoDefense,
    RefusalClassifier,
    RetrievalFilter,
    SystemPromptRule,
)
from .registry import REGISTRY, available, build_method, register
from .silentwall import CalibrationParams, SilentWallDefense

__all__ = [
    "ContainmentMethod",
    "BaseContainment",
    "EntityContext",
    "REGISTRY",
    "register",
    "build_method",
    "available",
    "NoDefense",
    "SystemPromptRule",
    "RetrievalFilter",
    "RefusalClassifier",
    "CleanReference",
    "LoraGradientAscent",
    "SilentWallDefense",
    "CalibrationParams",
]
