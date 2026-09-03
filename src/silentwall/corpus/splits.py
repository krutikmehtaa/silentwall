"""Dev and eval split assignment.

This is the module that decides whether the headline number is honest.

The reference defense is calibrated against a wall detector. If the same entities
appear in that calibration and in the reported evaluation, the reported AUC is
training loss wearing a costume. So the split is enforced by types, not by
convention: DevSplit physically cannot hand out eval entity ids, and asking for one
during a dev phase raises.

Splitting on pair_id rather than entity_id is stricter than it looks necessary. A
restricted entity and its matched control were chosen to resemble each other, so
putting one in dev and the other in eval would let a model learn the matching rule
instead of the behavioural signal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..errors import SplitLeakageError
from ..hashing import hash_obj, stable_seed
from ..types import Corpus, Entity, SplitName

__all__ = ["SplitAssignment", "DevSplit", "EvalSplit", "assign_splits"]


@dataclass(frozen=True, slots=True)
class _SplitView:
    """Shared behaviour for a one-sided view of the corpus."""

    name: SplitName
    pair_ids: frozenset[str]
    entities: tuple[Entity, ...]

    @property
    def entity_ids(self) -> frozenset[str]:
        return frozenset(e.entity_id for e in self.entities)

    @property
    def restricted(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.entity_class == "restricted")

    @property
    def deal_ids(self) -> tuple[str, ...]:
        return tuple(sorted({e.deal_id for e in self.entities if e.deal_id}))

    def contains(self, entity_id: str) -> bool:
        return entity_id in self.entity_ids

    def require(self, entity_id: str) -> None:
        """Raise if the entity is outside this split.

        Called by the runner on every generation request during a fit phase. This is
        the tripwire that turns a silent contamination bug into a loud failure.
        """
        if entity_id not in self.entity_ids:
            raise SplitLeakageError(
                f"entity {entity_id} is not in the {self.name} split. "
                f"A containment method tried to look at data it must not see during fit."
            )

    @property
    def ids_hash(self) -> str:
        return hash_obj(sorted(self.entity_ids))


@dataclass(frozen=True, slots=True)
class DevSplit(_SplitView):
    """Entities a containment method may fit on.

    Deliberately carries no reference to the eval side. There is no attribute on
    this object that could leak an eval entity id, so a method cannot reach one by
    accident even if it tries.
    """


@dataclass(frozen=True, slots=True)
class EvalSplit(_SplitView):
    """Entities the reported numbers are computed from."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    dev: DevSplit
    eval: EvalSplit
    corpus_hash: str
    split_seed: int

    @property
    def disjoint(self) -> bool:
        return not (self.dev.entity_ids & self.eval.entity_ids) and not (
            self.dev.pair_ids & self.eval.pair_ids
        )

    def audit(self) -> dict[str, object]:
        """Split audit for the run record, so a reader can verify the claim."""
        return {
            "dev_ids_hash": self.dev.ids_hash,
            "eval_ids_hash": self.eval.ids_hash,
            "splits_disjoint": self.disjoint,
            "n_dev_pairs": len(self.dev.pair_ids),
            "n_eval_pairs": len(self.eval.pair_ids),
            "corpus_hash": self.corpus_hash,
            "split_seed": self.split_seed,
        }

    def split_of(self, entity_id: str) -> SplitName:
        if self.dev.contains(entity_id):
            return "dev"
        if self.eval.contains(entity_id):
            return "eval"
        raise KeyError(f"entity {entity_id} is in neither split")


def assign_splits(corpus: Corpus, split_seed: int, dev_fraction: float = 0.5) -> SplitAssignment:
    """Partition matched pairs into dev and eval.

    Deterministic from the corpus hash and the seed, so the same corpus always
    splits the same way regardless of dictionary ordering or platform.
    """
    pair_ids = sorted({e.pair_id for e in corpus.entities})
    if len(pair_ids) < 2:
        raise SplitLeakageError(
            f"need at least 2 matched pairs to split, corpus has {len(pair_ids)}"
        )

    rng = random.Random(stable_seed(corpus.manifest.corpus_hash, split_seed))
    shuffled = list(pair_ids)
    rng.shuffle(shuffled)

    n_dev = max(1, min(len(shuffled) - 1, round(len(shuffled) * dev_fraction)))
    dev_pairs = frozenset(shuffled[:n_dev])
    eval_pairs = frozenset(shuffled[n_dev:])

    dev_entities = tuple(e for e in corpus.entities if e.pair_id in dev_pairs)
    eval_entities = tuple(e for e in corpus.entities if e.pair_id in eval_pairs)

    assignment = SplitAssignment(
        dev=DevSplit(name="dev", pair_ids=dev_pairs, entities=dev_entities),
        eval=EvalSplit(name="eval", pair_ids=eval_pairs, entities=eval_entities),
        corpus_hash=corpus.manifest.corpus_hash,
        split_seed=split_seed,
    )

    if not assignment.disjoint:
        raise SplitLeakageError("split produced overlapping dev and eval sets")
    return assignment
