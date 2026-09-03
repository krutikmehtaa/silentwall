"""Canonical serialization and content hashing.

Every hash in this system has to be stable across processes, machines and Python
versions, otherwise cache keys miss and reproducibility claims are false. That
means we cannot use repr() or the built-in hash(), and we cannot rely on dict
ordering. This module is the single place that decides what "the same value"
means.

Rules for canonical form:
  - mappings serialize with sorted keys
  - no insignificant whitespace
  - floats use repr, which round-trips exactly in Python 3
  - enums serialize as their value
  - sets serialize as sorted lists
  - tuples and lists both serialize as arrays
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

__all__ = [
    "canonical",
    "canonical_json",
    "sha256_hex",
    "hash_obj",
    "short_hash",
    "stable_seed",
]


def canonical(value: Any) -> Any:
    """Reduce a value to JSON-compatible primitives in canonical form."""
    if value is None or isinstance(value, bool | int | str):
        return value

    if isinstance(value, float):
        # repr round-trips floats exactly, str does too in py3 but repr is explicit
        return repr(value)

    if isinstance(value, Enum):
        return canonical(value.value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()

    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: canonical(getattr(value, f.name))
            for f in sorted(fields(value), key=lambda f: f.name)
        }

    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}

    if isinstance(value, frozenset | set):
        return sorted((canonical(v) for v in value), key=_sort_key)

    if isinstance(value, Sequence):
        return [canonical(v) for v in value]

    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _sort_key(value: Any) -> str:
    """Total order over canonical primitives, used only for set serialization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    """Canonical JSON text for a value."""
    return json.dumps(
        canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_hex(text: str) -> str:
    """Hex digest of a string, encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(*parts: Any) -> str:
    """Content hash over one or more values.

    Wrapping the parts in a list means hash_obj(a, b) and hash_obj([a, b]) differ,
    which avoids a class of accidental collision when callers pass sequences.
    """
    return sha256_hex(canonical_json(list(parts)))


def short_hash(*parts: Any, length: int = 12) -> str:
    """Truncated content hash, for ids that humans read."""
    return hash_obj(*parts)[:length]


def stable_seed(*parts: Any) -> int:
    """Derive a 63-bit integer seed from arbitrary values.

    Used wherever a random generator has to be reproducible from semantic inputs
    rather than from call order. 63 bits keeps it inside a positive int64, which
    numpy accepts without complaint.
    """
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
