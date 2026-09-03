"""Containment method registry.

Methods register themselves by decorator so adding one is a single file and no edit
to the harness. That is the substitutability property the design asks for, and it is
also what lets a test define a throwaway method and drive the whole pipeline with it.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ..errors import ConfigError

__all__ = ["REGISTRY", "register", "build_method", "available"]

REGISTRY: dict[str, type[Any]] = {}

T = TypeVar("T")


def register(cls: type[T]) -> type[T]:
    method_id = getattr(cls, "id", None)
    if not method_id or not isinstance(method_id, str):
        raise ConfigError(f"{cls.__name__} needs a non-empty string id to register")
    if method_id in REGISTRY and REGISTRY[method_id] is not cls:
        raise ConfigError(f"containment id {method_id!r} is already registered")
    REGISTRY[method_id] = cls
    return cls


def build_method(method_id: str, **params: Any) -> Any:
    if method_id not in REGISTRY:
        raise ConfigError(
            f"unknown containment method {method_id!r}, available: {sorted(REGISTRY)}"
        )
    return REGISTRY[method_id](**params)


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))
