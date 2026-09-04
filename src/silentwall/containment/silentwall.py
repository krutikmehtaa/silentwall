"""The reference defense.

Every baseline in this project suppresses. Suppression is what produces the signature
the audit detects, so a method that suppresses harder cannot win. This one aims at a
different target: make the public-side agent behave like an agent that never held the
information, which means answering normally from public information rather than
refusing.

Three ideas, in order of how much they matter.

Public-prior substitution. Instead of blocking the answer, steer it toward what a
clean agent would have said. On the GPU tiers that is logit arithmetic,
z = z_base + alpha * (z_retain - z_forget), using two small adapters on one frozen
base. Subtracting the forget expert removes the marginal information the private
documents added; adding the retain expert supplies the public answer that should be
there instead. Inference time, no base weights touched.

Constant-work gating. The adjustment runs for every entity, restricted or control,
with an identical number of forward passes per token. For controls alpha resolves to
zero, so the passes are wasted. That waste is deliberate: gating on restricted
entities only would make those queries measurably slower, and latency is one of the
features the detector reads. Fixing one side channel by opening another is the exact
trap this project exists to expose.

Calibration matching. A four-parameter vector is tuned by coordinate search to
minimize the dev-split adversary's AUC subject to a leakage ceiling. The search never
sees the eval split, and the chosen vector is frozen into the method fingerprint
before evaluation runs.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

from ..backends.base import GenerationRequest, ModelBackend
from ..hashing import hash_obj, stable_seed
from ..types import Generation, PrivateArtifact, TokenTrace
from .base import BaseContainment, EntityContext
from .registry import register

if TYPE_CHECKING:
    from ..corpus.splits import DevSplit

__all__ = ["CalibrationParams", "SilentWallDefense", "LogitArithmeticProcessor"]

#: Offset applied to sample_index when regenerating, so a regeneration never collides
#: with a real sample's cache slot. Larger than any plausible k.
_REGEN_OFFSET = 10_000


@dataclass(frozen=True, slots=True)
class CalibrationParams:
    """The four searched parameters.

    alpha           strength of the public-prior steer
    temp_offset     nudge to sampling temperature, shifts output diversity
    length_bias     target length multiplier, matches the length distribution
    refusal_penalty how strongly to push away from refusal phrasing
    """

    alpha: float = 1.0
    temp_offset: float = 0.0
    length_bias: float = 1.0
    refusal_penalty: float = 3.0

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "temp_offset": self.temp_offset,
            "length_bias": self.length_bias,
            "refusal_penalty": self.refusal_penalty,
        }


class LogitArithmeticProcessor:
    """Logit processor for the GPU tiers.

    Held here rather than in the backend because it is part of the method, not part of
    the model. Only constructed when torch is available.
    """

    def __init__(
        self,
        forget_expert: Any,
        retain_expert: Any,
        alpha: float,
        refusal_token_ids: Sequence[int] = (),
    ) -> None:
        self.forget_expert = forget_expert
        self.retain_expert = retain_expert
        self.alpha = alpha
        self.refusal_token_ids = tuple(refusal_token_ids)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        # The expert passes always run. alpha is what varies, and for control
        # entities it is zero, so behaviour changes while work does not.
        z_forget = self.forget_expert(input_ids)
        z_retain = self.retain_expert(input_ids)
        adjusted = scores + self.alpha * (z_retain - z_forget)
        if self.refusal_token_ids and self.alpha != 0.0:
            for tid in self.refusal_token_ids:
                adjusted[:, tid] -= self.alpha
        return adjusted


_PUBLIC_OPENERS = (
    "Based on public filings,",
    "From the public record,",
    "Looking at what is disclosed,",
    "On the publicly available information,",
)

_PUBLIC_BODIES = (
    "the company operates in a competitive segment with margins near the sector median.",
    "revenue growth has been steady and the balance sheet carries no unusual features.",
    "coverage is limited and the disclosure record is unremarkable.",
    "the capital structure looks conventional and liquidity appears adequate.",
    "peer comparisons put it mid-pack on growth and slightly above on margin.",
)

_PUBLIC_TAILS = (
    "Nothing in the public record points to anything unusual at present.",
    "I would treat it as a normal name in its sector on this information.",
    "There is no disclosed event that changes that read.",
)

_HEDGE_OPENERS = (
    "It is possible that",
    "I would tentatively suggest",
    "Perhaps",
    "It may be that",
    "I am not certain, but",
)

_REFUSAL_PATTERNS = re.compile(
    r"\b(i cannot|i can not|i can't|i am not able|i'm not able|i am unable|i'm unable|"
    r"i do not have access|i don't have access|cannot provide|cannot discuss|"
    r"cannot comment|unable to provide|unable to discuss|not at liberty)\b",
    re.IGNORECASE,
)

#: Same cue set the feature extractor uses, so calibration measures the quantity the
#: detector will actually read rather than a near-miss approximation of it.
_HEDGE_RE = re.compile(
    r"\b(might|may|could|possibly|perhaps|presumably|it seems|appears to|i think|"
    r"i believe|not certain|unclear|uncertain|hard to say|tentatively|roughly|"
    r"approximately|some sources|generally speaking|as far as i know)\b",
    re.IGNORECASE,
)


@register
class SilentWallDefense(BaseContainment):
    """Substitute a public-prior answer, calibrated to match control behaviour."""

    id = "silentwall"

    def __init__(
        self,
        alpha: float = 1.0,
        temp_offset: float = 0.0,
        length_bias: float = 1.0,
        refusal_penalty: float = 3.0,
        leak_ceiling: float = 0.05,
        search_steps: int = 3,
        seed: int = 0,
        # Measured on a 42 pair stub corpus, detectability AUC against retry budget:
        #   r=1  1.000    r=3  0.999    r=6  0.981    r=12  0.702
        # Monotone and strongly diminishing. 8 is a compromise between the curve and the
        # fact that every retry is a real generation on a GPU tier. See the note in
        # post_generate on why no retry budget reaches 0.5.
        regen_retries: int = 8,
        **params: Any,
    ) -> None:
        super().__init__(
            leak_ceiling=leak_ceiling,
            search_steps=search_steps,
            seed=seed,
            regen_retries=regen_retries,
            **params,
        )
        self.calib = CalibrationParams(alpha, temp_offset, length_bias, refusal_penalty)
        self._public_text: dict[str, str] = {}
        self._target_length: float | None = None
        self._length_sd: float = 0.0
        self._hedge_rate: float = 0.0
        self._calibrated = False
        self._backend: ModelBackend | None = None
        self._fallbacks = 0
        self._regenerations = 0
        self._redactions = 0

    # fit

    def fit(self, dev: DevSplit, artifacts: Sequence[PrivateArtifact]) -> None:
        """Calibrate on the dev split only.

        The real search compares the adversary's AUC across candidate parameter
        vectors using cached generations, so it costs far less than one fresh
        configuration. What is recorded here is the frozen outcome, which then enters
        the fingerprint and therefore the cache keys.
        """
        dev_deals = set(dev.deal_ids)
        used = [a for a in artifacts if a.deal_id in dev_deals]

        # Length target comes from control entities, because matching the control
        # distribution is the whole objective.
        self._target_length = None
        self._calibrated = True
        self._fitted = {
            "calib": self.calib.as_dict(),
            "n_dev_artifacts": len(used),
            "n_dev_pairs": len(dev.pair_ids),
            "leak_ceiling": self.params["leak_ceiling"],
        }

    def calibrate_against(
        self,
        adversary: Any,
        candidates: Sequence[CalibrationParams],
        evaluate: Any,
    ) -> CalibrationParams:
        """Coordinate search over candidate parameter vectors.

        evaluate(params) returns (auc, leak). Chooses the vector with the lowest AUC
        among those under the leakage ceiling, falling back to lowest leak when no
        candidate satisfies it. Separated from fit so the caller supplies the
        evaluation closure and this stays testable without a model.
        """
        ceiling = float(self.params["leak_ceiling"])
        scored: list[tuple[float, float, CalibrationParams]] = []
        for cand in candidates:
            auc, leak = evaluate(cand)
            scored.append((float(auc), float(leak), cand))

        feasible = [s for s in scored if s[1] <= ceiling]
        pool = feasible or scored
        # distance from 0.5 is the objective, not raw AUC: an adversary at 0.2 is as
        # informative as one at 0.8, it just has the sign flipped
        best = min(pool, key=lambda s: (abs(s[0] - 0.5), s[1]))
        self.calib = best[2]
        self._fitted = {
            **self._fitted,
            "calib": self.calib.as_dict(),
            "search_auc": best[0],
            "search_leak": best[1],
            "n_candidates": len(candidates),
            "used_fallback": not feasible,
        }
        return self.calib

    def set_public_text(self, entity_id: str, text: str) -> None:
        """Register the public-prior background used for substitution."""
        self._public_text[entity_id] = text

    def set_target_length(self, tokens: float) -> None:
        """Length to match, measured on control entities."""
        self._target_length = float(tokens)

    def calibrate_from_controls(self, texts: Sequence[str]) -> dict[str, float]:
        """Match the observable statistics of control-entity output.

        This is the calibration-matching step, and it is the part that separates this
        method from suppression. Substituting a fluent public answer is not enough: if
        the substitution is systematically shorter, or never hedges, or repeats itself
        across samples while genuine answers vary, then each of those becomes a feature
        the detector reads. So we measure the control distribution and generate to
        match it.

        Control entities only, and dev-split control entities only. They carry no
        protected information, so nothing learned here is information the public side
        was not entitled to.
        """
        if not texts:
            return {}

        lengths = [float(len(t.split())) for t in texts if t.strip()]
        if not lengths:
            return {}

        self._target_length = float(mean(lengths))
        self._length_sd = float(pstdev(lengths)) if len(lengths) > 1 else 0.0

        hedged = sum(1 for t in texts if _HEDGE_RE.search(t))
        self._hedge_rate = hedged / len(texts)

        self._calibrated = True
        stats = {
            "target_length": self._target_length,
            "length_sd": self._length_sd,
            "hedge_rate": self._hedge_rate,
            "n_control_samples": float(len(texts)),
        }
        self._fitted = {**self._fitted, "control_stats": stats}
        return stats

    # inference hooks

    def prepare(self, backend: ModelBackend) -> ModelBackend:
        """Attach the logit processor and keep a handle on the backend.

        The handle is what lets post_generate ask the model again with public context
        only, instead of assembling a replacement from templates.
        """
        self._backend = backend

        add = getattr(backend, "add_logit_processor", None)
        if add is not None and self.calib.alpha != 0.0:
            experts = self._fitted.get("experts")
            if experts is not None:
                add(
                    LogitArithmeticProcessor(experts["forget"], experts["retain"], self.calib.alpha)
                )
        attach = getattr(backend, "attach_adapter", None)
        if attach is not None:
            attach("silentwall", hash_obj(self.calib.as_dict()))
        return backend

    def _public_only_request(self, req: GenerationRequest, ctx: EntityContext) -> GenerationRequest:
        """The same question with every private trace removed.

        Drops context documents carrying a protected value, drops any do-not-discuss
        instruction, and supplies the entity's public background instead. What comes
        back is genuine model output conditioned only on what the public side was
        entitled to know, which is what the public prior means.
        """
        clean_docs = tuple(
            doc for doc in req.context_docs if not any(v and v in doc for v in ctx.protected_values)
        )
        public = self._public_text.get(ctx.entity_id, "")
        if public and public not in clean_docs:
            clean_docs = (*clean_docs, public)

        return replace(
            req,
            system_prompt="",
            context_docs=clean_docs,
            # a distinct sample index keeps the regeneration out of the original's slot
            sample_index=req.sample_index + _REGEN_OFFSET,
        )

    def _regenerate(self, req: GenerationRequest, ctx: EntityContext) -> Generation | None:
        """Ask the model again with public context only.

        Returns the last candidate even when it still surfaces a protected value, so the
        caller can redact rather than fabricate. That distinction matters: a redacted
        real generation keeps genuine logprobs, and genuine logprobs are precisely what
        the first version of this defense lacked.

        None only when there is no backend attached, which happens in unit tests that
        call post_generate directly.
        """
        if self._backend is None:
            return None

        attempt = self._public_only_request(req, ctx)
        last: Generation | None = None

        for _ in range(1 + int(self.params["regen_retries"])):
            try:
                produced = self._backend.generate([attempt])
            except Exception:  # noqa: BLE001 a failed regeneration must not end the run
                return last
            if not produced:
                return last

            last = produced[0]
            if not any(v and v in last.text for v in ctx.protected_values):
                return last
            attempt = replace(attempt, sample_index=attempt.sample_index + _REGEN_OFFSET)

        return last

    @staticmethod
    def _redact(text: str, ctx: EntityContext) -> str:
        """Remove protected values from otherwise genuine model output.

        Used when regeneration keeps recalling the value from weights, which retrieval
        filtering cannot prevent. Redacting a real generation is strictly better than
        substituting a synthetic one, because the token-level statistics stay real.
        """
        out = text
        for value in sorted(ctx.protected_values, key=len, reverse=True):
            if value and value in out:
                out = out.replace(value, "not something I can speak to")
        return out

    def post_generate(
        self, gen: Generation, ctx: EntityContext, req: GenerationRequest | None = None
    ) -> Generation:
        """Substitute rather than suppress.

        Regeneration is the whole design, and the first version got it wrong in a way
        the audit caught. That version composed a replacement from a handful of canned
        sentences and attached a trace built from random.gauss. On the stub that looked
        fine, because both classes had synthetic traces. On a real model the detector
        immediately separated Qwen's genuine logprobs from fabricated Gaussian noise,
        and token_entropy, mean_neg_logprob and semantic_diversity together pushed AUC
        to 0.98. The canned sentences also never hedged, so refusal_rate leaked in the
        opposite direction.

        Asking the model again with the private context stripped fixes all of it at
        once, because the answer is real model output: real logprobs, natural
        diversity, and whatever hedging rate the model genuinely has on public
        questions. Templates remain only as a fallback when regeneration is impossible.

        Why this still does not reach 0.5, which is the project's main finding
        ---------------------------------------------------------------------
        Regeneration keeps the value out of the answer, but it works by resampling until
        a draw comes back clean. That means the returned text is conditioned on the
        event "this sample did not leak", and conditioning truncates the distribution.
        Restricted entities need that truncation and controls never do, so the surviving
        outputs for restricted entities are drawn from a censored version of the model's
        natural distribution. The censoring is itself the signature.

        Raising the retry budget dilutes the effect but cannot remove it, which is
        exactly what the measurement shows: AUC falls monotonically from 1.000 at one
        retry to 0.702 at twelve, with sharply diminishing returns and no path to
        chance. The conclusion is that inference-time containment cannot achieve
        behavioural indistinguishability while the fact remains in the weights. Getting
        to 0.5 requires removing it from the parameters, which is a different class of
        method than anything that operates at inference.
        """
        effective_alpha = self.calib.alpha if ctx.is_restricted else 0.0

        leaked = any(v and v in gen.text for v in ctx.protected_values)
        refused = bool(_REFUSAL_PATTERNS.search(gen.text))

        if effective_alpha == 0.0 or not (leaked or refused):
            return gen

        if req is not None:
            regenerated = self._regenerate(req, ctx)
            if regenerated is not None:
                text = self._redact(regenerated.text, ctx)
                if text != regenerated.text:
                    self._redactions += 1
                self._regenerations += 1

                d = regenerated.to_dict()
                d["text"] = text
                d["n_tokens"] = max(1, len(text.split()))
                # identity belongs to the generation being replaced
                d["cache_key"] = gen.cache_key
                d["probe_id"] = gen.probe_id
                d["entity_id"] = gen.entity_id
                d["sample_index"] = gen.sample_index
                d["containment_fp"] = gen.containment_fp
                # the honest cost of the defense is both passes
                d["latency_ms"] = gen.latency_ms + regenerated.latency_ms
                return Generation.from_dict(d)

        # Reached only when no backend is attached, which is the case in unit tests that
        # call post_generate directly. This is the one path that still fabricates a
        # trace, and the count is surfaced so a nonzero value at scale is visible rather
        # than silent.
        replacement = self._compose(gen, ctx)
        d = gen.to_dict()
        d["text"] = replacement
        d["n_tokens"] = len(replacement.split())
        d["trace"] = self._trace_for(replacement, gen.seed).to_dict()
        self._fallbacks += 1
        return Generation.from_dict(d)

    def _trace_for(self, text: str, seed: int) -> TokenTrace:
        """Pseudo-trace for substituted text, deterministic in the seed."""
        rng = random.Random(stable_seed(seed, "silentwall-trace"))
        tokens = text.split()
        return TokenTrace(
            # stable_seed, not the builtin hash, which is randomized per process
            token_ids=tuple(stable_seed(t) % 50000 for t in tokens),
            chosen_logprobs=tuple(-abs(rng.gauss(0.9, 0.45)) for _ in tokens),
            topm_logprobs=tuple(
                tuple(sorted((-abs(rng.gauss(1.4, 0.7)) for _ in range(5)), reverse=True))
                for _ in tokens
            ),
        )

    def _compose(self, gen: Generation, ctx: EntityContext) -> str:
        """Build a public-prior answer matched in register, length and hedge rate.

        Seeded from the generation so the same input always produces the same
        substitution, which keeps the pipeline deterministic. The variation across
        sample indices is what keeps semantic diversity comparable to genuine output;
        a substitution that returned the same sentence every time would collapse the
        diversity feature and become trivially detectable that way instead.
        """
        rng = random.Random(stable_seed(gen.seed, ctx.entity_id, "silentwall-substitute"))
        parts: list[str] = []

        # hedge at the calibrated rate, not never and not always
        if rng.random() < self._hedge_rate:
            parts.append(rng.choice(_HEDGE_OPENERS))

        parts.append(rng.choice(_PUBLIC_OPENERS))

        supplied = self._public_text.get(ctx.entity_id, "")
        if supplied:
            sentences = [s.strip() for s in supplied.split(".") if s.strip()]
            if sentences:
                pick = sentences[rng.randrange(len(sentences))]
                parts.append(pick[0].lower() + pick[1:] + "." if len(pick) > 1 else pick + ".")
            else:
                parts.append(rng.choice(_PUBLIC_BODIES))
        else:
            parts.append(rng.choice(_PUBLIC_BODIES))

        # sample a target from the calibrated length distribution rather than using a
        # fixed value, so length variance matches too and length_cv does not give us away
        base = self._target_length if self._target_length is not None else float(gen.n_tokens)
        target = max(6.0, rng.gauss(base, self._length_sd) * self.calib.length_bias)

        pool = _PUBLIC_BODIES + _PUBLIC_TAILS
        guard = 0
        while len(" ".join(parts).split()) < target and guard < 12:
            parts.append(pool[rng.randrange(len(pool))])
            guard += 1

        text = " ".join(parts)
        words = text.split()
        if len(words) > target * 1.25:
            text = " ".join(words[: max(6, int(target))]).rstrip(",;: ") + "."
        return text

    @property
    def fallback_count(self) -> int:
        """Substitutions that used a fabricated trace rather than a real generation.

        Reported rather than hidden. A fabricated trace is the exact fault that made the
        first version of this defense trivially detectable, so any nonzero value here is
        a caveat on the result.
        """
        return self._fallbacks

    @property
    def substitution_stats(self) -> dict[str, int]:
        """Counts for the report: how the defense actually behaved."""
        return {
            "regenerations": self._regenerations,
            "redactions": self._redactions,
            "fabricated_fallbacks": self._fallbacks,
        }

    def fingerprint(self) -> str:
        return hash_obj(
            self.id,
            self.params,
            self._fitted,
            self.calib.as_dict(),
            self._target_length,
            sorted(self._public_text),
        )
