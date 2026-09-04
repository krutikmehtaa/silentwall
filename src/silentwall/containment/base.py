"""Containment method interface.

One interface, five hooks. Every method from "no defense" through the reference
defense plugs in here, and the evaluation harness never branches on which one is
active. That property is what makes the comparison across methods fair: if the
harness treated one method specially, any difference in its numbers could be the
harness rather than the method.

The hooks map onto where a barrier can be enforced:

  fit                calibrate on dev-split entities only
  prepare            wrap or modify the backend
  transform_request  change what the model is asked
  filter_context     change what the model is shown
  post_generate      change what the model's output becomes

EntityContext is passed to containment hooks and never to scoring. A method is
allowed to know it is looking at a restricted entity. The detector is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..backends.base import GenerationRequest, ModelBackend
from ..hashing import hash_obj
from ..types import EntityClass, Generation, PrivateArtifact

if TYPE_CHECKING:
    from ..corpus.splits import DevSplit
    from ..probes.toolenv import RetrievedDoc

__all__ = ["EntityContext", "ContainmentMethod", "BaseContainment"]


@dataclass(frozen=True, slots=True)
class EntityContext:
    """What a containment method is allowed to know about the entity in play."""

    entity_id: str
    display_name: str
    entity_class: EntityClass
    protected_field_ids: tuple[str, ...] = ()
    protected_values: tuple[str, ...] = ()
    deal_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_restricted(self) -> bool:
        return self.entity_class == "restricted"


@runtime_checkable
class ContainmentMethod(Protocol):
    id: str
    requires_clean_substrate: bool

    def fit(self, dev: DevSplit, artifacts: Sequence[PrivateArtifact]) -> None: ...
    def prepare(self, backend: ModelBackend) -> ModelBackend: ...
    def transform_request(
        self, req: GenerationRequest, ctx: EntityContext
    ) -> GenerationRequest: ...
    def filter_context(
        self, docs: Sequence[RetrievedDoc], ctx: EntityContext
    ) -> Sequence[RetrievedDoc]: ...
    def post_generate(
        self, gen: Generation, ctx: EntityContext, req: GenerationRequest | None = None
    ) -> Generation: ...
    def fingerprint(self) -> str: ...


class BaseContainment:
    """Identity behaviour for every hook. Subclasses override what they need.

    Defaults matter here: a method that only touches one pathway should not have to
    write four pass-through methods, and more importantly should not accidentally
    change a pathway it did not intend to.
    """

    id: str = "base"
    requires_clean_substrate: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self._fitted: dict[str, Any] = {}

    def fit(self, dev: DevSplit, artifacts: Sequence[PrivateArtifact]) -> None:
        return None

    def prepare(self, backend: ModelBackend) -> ModelBackend:
        return backend

    def transform_request(self, req: GenerationRequest, ctx: EntityContext) -> GenerationRequest:
        return req

    def filter_context(
        self, docs: Sequence[RetrievedDoc], ctx: EntityContext
    ) -> Sequence[RetrievedDoc]:
        return docs

    def post_generate(
        self, gen: Generation, ctx: EntityContext, req: GenerationRequest | None = None
    ) -> Generation:
        return gen

    def fingerprint(self) -> str:
        """Identity of this method plus its parameters plus anything learned in fit.

        Goes into every cache key, so refitting with different parameters cannot
        reuse generations from the previous configuration.
        """
        return hash_obj(self.id, self.params, self._fitted)
