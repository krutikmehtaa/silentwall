"""Deterministic backend that needs no model weights.

This exists so the entire pipeline, including feature extraction, detector fitting
and bootstrap intervals, runs on CPU in seconds. That is what makes the test suite
runnable in CI and what lets you validate the plumbing before spending a GPU hour.

It is not a toy. To be useful as a test double it has to reproduce the phenomenon
under study, so it behaves like a plausible instruction-following model:

  - It carries simulated parametric knowledge, meaning it can recall a protected
    value even when that value is absent from its context. That is what makes
    retrieval filtering an interesting baseline rather than a trivially perfect one.
  - It reads context and memory, so it leaks when private documents are in scope.
  - It obeys a refusal instruction, which is what produces the abstention signature
    the audit is designed to catch.

None of that is hardcoded per method. The behaviour falls out of what each
containment method does to the prompt, which is the point.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence

from ..hashing import hash_obj, stable_seed
from ..types import Generation, TokenTrace
from .base import BaseBackend, GenerationRequest

__all__ = ["StubBackend"]

_REFUSAL_CUES = (
    "do not discuss",
    "must not discuss",
    "do not reveal",
    "confidential and must not",
    "you are prohibited",
)

_REFUSALS = (
    "I cannot discuss that topic.",
    "I am not able to comment on this company.",
    "I do not have access to information about that.",
    "I am unable to provide details here.",
)

_HEDGES = (
    "It is possible that",
    "I would tentatively suggest",
    "Perhaps",
    "It may be that",
    "I am not certain, but",
)

_NEUTRAL_OPENERS = (
    "Based on public filings,",
    "From the public record,",
    "Looking at what is disclosed,",
    "On the publicly available information,",
)

_NEUTRAL_BODIES = (
    "the company operates in a competitive segment with margins near the sector median.",
    "revenue growth has been steady and the balance sheet carries no unusual features.",
    "coverage is limited and the disclosure record is unremarkable.",
    "the capital structure looks conventional and liquidity appears adequate.",
    "peer comparisons put it mid-pack on growth and slightly above on margin.",
)

_TOOL_HINTS = ("search_docs", "calc")


class StubBackend(BaseBackend):
    """Seeded pseudo-model.

    Parameters
    ----------
    memorized:
        entity_id to the protected value strings the model has absorbed into its
        weights. Recalled with probability ``recall_rate`` even with no context.
    recall_rate:
        how often parametric knowledge surfaces unprompted.
    context_leak_rate:
        how often a value present in the context gets repeated back.
    """

    tier = "stub"
    model_id = "stub://deterministic"

    def __init__(
        self,
        memorized: Mapping[str, Sequence[str]] | None = None,
        recall_rate: float = 0.55,
        context_leak_rate: float = 0.9,
        refusal_compliance: float = 0.85,
        latency_base_ms: float = 4.0,
    ) -> None:
        super().__init__()
        self.memorized: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (memorized or {}).items()
        }
        self.recall_rate = recall_rate
        self.context_leak_rate = context_leak_rate
        self.refusal_compliance = refusal_compliance
        self.latency_base_ms = latency_base_ms
        self.set_fingerprint_extra("recall_rate", recall_rate)
        self.set_fingerprint_extra("context_leak_rate", context_leak_rate)
        # hash_obj, never the builtin hash(). Python randomizes string hashing per
        # process, so a builtin hash here would give the backend a different
        # fingerprint on every run, every cache key would miss, and a multi-session
        # sweep would regenerate everything from scratch.
        self.set_fingerprint_extra(
            "memorized", hash_obj({k: list(v) for k, v in sorted(self.memorized.items())})
        )

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        return [self._one(r) for r in requests]

    def _one(self, req: GenerationRequest) -> Generation:
        started = time.perf_counter()
        rng = random.Random(req.seed)
        prompt = req.full_prompt()
        lowered = prompt.lower()

        instructed_to_refuse = any(cue in lowered for cue in _REFUSAL_CUES)
        tool_calls: list[str] = []

        if instructed_to_refuse and rng.random() < self.refusal_compliance:
            text = rng.choice(_REFUSALS)
        else:
            text = self._answer(req, rng, prompt, tool_calls)

        n_tokens = max(1, len(text.split()))
        trace = self._trace(text, rng)

        # Simulated latency scales with output length, the way a real decode does.
        elapsed = (time.perf_counter() - started) * 1000.0
        latency = self.latency_base_ms + 0.35 * n_tokens + elapsed

        return Generation(
            cache_key="",  # filled in by the cache layer
            probe_id=req.probe_id,
            entity_id=req.entity_id,
            sample_index=req.sample_index,
            seed=req.seed,
            text=text,
            n_tokens=n_tokens,
            backend_fp=self.fingerprint(),
            containment_fp="",
            trace=trace,
            tool_calls=tuple(tool_calls),
            latency_ms=latency,
            latency_trustworthy=True,
        )

    def _answer(
        self,
        req: GenerationRequest,
        rng: random.Random,
        prompt: str,
        tool_calls: list[str],
    ) -> str:
        parts: list[str] = []

        if req.tools_exposed:
            for hint in _TOOL_HINTS:
                if hint in req.tools_exposed and rng.random() < 0.8:
                    tool_calls.append(hint)

        # values visible in the context or memory block
        context_blob = "\n".join(req.context_docs)
        visible = [v for v in self.memorized.get(req.entity_id, ()) if v and v in context_blob]

        revealed: list[str] = []
        for value in visible:
            if rng.random() < self.context_leak_rate:
                revealed.append(value)

        # parametric recall, independent of what is in the context
        if not revealed:
            for value in self.memorized.get(req.entity_id, ()):
                if rng.random() < self.recall_rate:
                    revealed.append(value)
                    break

        if rng.random() < 0.25:
            parts.append(rng.choice(_HEDGES))

        parts.append(rng.choice(_NEUTRAL_OPENERS))

        if revealed:
            parts.append(
                f"the figure associated with this name is {revealed[0]}, "
                f"which is the number recorded in the working materials."
            )
        else:
            parts.append(rng.choice(_NEUTRAL_BODIES))

        if rng.random() < 0.35:
            parts.append(rng.choice(_NEUTRAL_BODIES))

        return " ".join(parts)

    def _trace(self, text: str, rng: random.Random) -> TokenTrace:
        """Plausible logprobs so entropy features have something to read.

        Values are not meaningful as probabilities. They only need to be
        deterministic and to vary with the text, so the feature extractor and the
        detector get exercised.
        """
        tokens = text.split()
        ids = tuple(stable_seed(t) % 50000 for t in tokens)
        chosen = tuple(-abs(rng.gauss(0.9, 0.45)) for _ in tokens)
        topm = tuple(
            tuple(sorted((-abs(rng.gauss(1.4, 0.7)) for _ in range(5)), reverse=True))
            for _ in tokens
        )
        return TokenTrace(token_ids=ids, chosen_logprobs=chosen, topm_logprobs=topm)
