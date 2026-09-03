"""Cache key derivation.

The key has to cover everything that could change the output and nothing that
cannot. Miss a component and you silently serve a stale generation from a different
configuration, which is the worst kind of bug here because the numbers still look
reasonable. Include something irrelevant, like a directory path, and every Kaggle
session looks like a fresh configuration and the weekly quota evaporates.
"""

from __future__ import annotations

from ..backends.base import GenerationRequest
from ..hashing import hash_obj

__all__ = ["cache_key"]


def cache_key(backend_fp: str, containment_fp: str, request: GenerationRequest) -> str:
    """Content hash over the full identity of a generation request."""
    return hash_obj(
        {
            "backend": backend_fp,
            "containment": containment_fp,
            **request.cache_components(),
        }
    )
