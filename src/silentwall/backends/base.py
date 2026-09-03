"""Model backend interface.

One protocol, several implementations, chosen by config. The reason this abstraction
earns its keep is the cost constraint: the same pipeline has to run on a CPU with no
weights for tests, on a 1.5B model for iteration, and on a 7B in 4-bit for the
reported numbers. Swapping between those is a config change, not a code change.

fingerprint() is the part that matters for correctness. It goes into every cache key,
so attaching a different adapter cannot silently reuse another configuration's
generations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config import SamplingConfig
from ..hashing import hash_obj, stable_seed
from ..types import Generation

__all__ = ["GenerationRequest", "ModelBackend", "derive_seed", "make_backend"]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One sampled completion to produce."""

    prompt: str
    sampling: SamplingConfig
    sample_index: int
    probe_id: str = ""
    entity_id: str = ""
    tool_state_hash: str = ""
    want_logprobs: bool = True
    system_prompt: str = ""
    context_docs: tuple[str, ...] = ()
    tools_exposed: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def seed(self) -> int:
        return derive_seed(self.sampling.base_seed, self.full_prompt(), self.sample_index)

    def full_prompt(self) -> str:
        """The exact text the model sees. Part of the cache key, so it must be stable."""
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"[system]\n{self.system_prompt}")
        if self.context_docs:
            parts.append("[context]\n" + "\n---\n".join(self.context_docs))
        if self.tools_exposed:
            parts.append("[tools available] " + ", ".join(sorted(self.tools_exposed)))
        parts.append(f"[user]\n{self.prompt}")
        return "\n\n".join(parts)

    def cache_components(self) -> dict[str, Any]:
        s = self.sampling
        return {
            "prompt": self.full_prompt(),
            "tool_state": self.tool_state_hash,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "max_new_tokens": s.max_new_tokens,
            "topm_logprobs": s.topm_logprobs if self.want_logprobs else 0,
            "sample_index": self.sample_index,
            "seed": self.seed,
        }


def derive_seed(base_seed: int, prompt: str, sample_index: int) -> int:
    """Per-sample seed derived from meaning, not from call order.

    Sample 0 is always the same sample. That is what makes leak@1 a well defined
    quantity rather than whichever generation happened to land first.
    """
    return stable_seed(base_seed, prompt, sample_index)


@runtime_checkable
class ModelBackend(Protocol):
    tier: str
    model_id: str

    def fingerprint(self) -> str: ...
    def supports_logprobs(self) -> bool: ...
    def generate(self, requests: Sequence[GenerationRequest]) -> list[Generation]: ...


class BaseBackend:
    """Shared fingerprint and adapter bookkeeping."""

    tier: str = "stub"
    model_id: str = "stub://deterministic"

    def __init__(self) -> None:
        self._adapters: tuple[str, ...] = ()
        self._extra: dict[str, Any] = {}

    def attach_adapter(self, name: str, content_hash: str) -> None:
        """Record an adapter so it lands in the fingerprint and therefore in cache keys."""
        self._adapters = tuple(sorted({*self._adapters, f"{name}:{content_hash}"}))

    def set_fingerprint_extra(self, key: str, value: Any) -> None:
        self._extra[key] = value

    def fingerprint(self) -> str:
        return hash_obj(self.tier, self.model_id, self._adapters, self._extra)

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        raise NotImplementedError


def make_backend(tier: str, model_id: str | None = None, **kw: Any) -> ModelBackend:
    """Construct a backend for a tier. Import of torch is deferred to here."""
    if tier == "stub":
        from .stub import StubBackend

        return StubBackend(**kw)

    from .hf import HfBackend

    return HfBackend(tier=tier, model_id=model_id, **kw)
