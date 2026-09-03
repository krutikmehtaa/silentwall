"""Generation execution.

Assembles the prompt for a probe, applies the active containment method's hooks,
checks the cache, generates what is missing, writes it down before using it, and
marks the unit complete.

The split tripwire lives here. During a fit phase every request carries the dev split,
and asking for an entity outside it raises SplitLeakageError rather than quietly
contaminating the calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..backends.base import GenerationRequest, ModelBackend
from ..cache.keys import cache_key
from ..cache.store import LayeredCache
from ..config import SilentwallConfig
from ..containment.base import ContainmentMethod, EntityContext
from ..corpus.splits import DevSplit
from ..errors import BackendOOMError
from ..probes.memory import MemoryStore
from ..probes.toolenv import RetrievedDoc, ToolEnv
from ..types import Corpus, Entity, Generation, Probe
from .checkpoint import Checkpoint, NullCheckpoint
from .plan import WorkUnit

__all__ = ["ExecutionContext", "ExecutionStats", "execute_units", "build_request"]


@dataclass
class ExecutionStats:
    from_cache: int = 0
    generated: int = 0
    deferred: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.from_cache + self.generated


@dataclass
class ExecutionContext:
    """Everything a generation needs that is not the probe itself."""

    corpus: Corpus
    cfg: SilentwallConfig
    backend: ModelBackend
    method: ContainmentMethod
    cache: LayeredCache
    tool_env: ToolEnv | None = None
    memory: MemoryStore | None = None
    public_text: dict[str, str] = field(default_factory=dict)
    dev_guard: DevSplit | None = None

    def entity(self, entity_id: str) -> Entity:
        return self.corpus.entity(entity_id)

    def context_for(self, entity_id: str) -> EntityContext:
        ent = self.entity(entity_id)
        deal = self.corpus.deal_for_entity(entity_id)
        fields = deal.protected_fields if deal else ()
        return EntityContext(
            entity_id=ent.entity_id,
            display_name=ent.display_name,
            entity_class=ent.entity_class,
            protected_field_ids=tuple(f.field_id for f in fields),
            protected_values=tuple(f.value_raw for f in fields)
            + tuple(f.value_normalized for f in fields),
            deal_id=ent.deal_id,
        )


def _retrieve(ctx: ExecutionContext, probe: Probe, ectx: EntityContext) -> list[RetrievedDoc]:
    """Gather context documents from the tool store and the memory store."""
    docs: list[RetrievedDoc] = []

    if ctx.tool_env is not None:
        query = f"{ectx.display_name} {probe.prompt}"
        docs.extend(ctx.tool_env.store.search(query, top_k=3))

    if probe.memory_primed and ctx.memory is not None:
        for entry in ctx.memory.read(probe.entity_id):
            docs.append(
                RetrievedDoc(
                    doc_id=f"mem:{entry.entry_id}",
                    text=entry.text,
                    score=1.0,
                    entity_id=entry.entity_id,
                    is_private=entry.is_private,
                )
            )

    public = ctx.public_text.get(probe.entity_id)
    if public:
        docs.append(RetrievedDoc(f"pub:{probe.entity_id}", public, 0.5, probe.entity_id, False))

    return docs


def build_request(
    ctx: ExecutionContext,
    probe: Probe,
    sample_index: int,
) -> tuple[GenerationRequest, EntityContext]:
    """Assemble one request with all containment hooks applied."""
    ectx = ctx.context_for(probe.entity_id)

    docs = _retrieve(ctx, probe, ectx)
    docs = list(ctx.method.filter_context(docs, ectx))

    tool_hash = ""
    if probe.tools_exposed and ctx.tool_env is not None:
        tool_hash = ctx.tool_env.state_hash(probe.tools_exposed)

    req = GenerationRequest(
        prompt=probe.prompt,
        sampling=ctx.cfg.sampling,
        sample_index=sample_index,
        probe_id=probe.probe_id,
        entity_id=probe.entity_id,
        tool_state_hash=tool_hash,
        want_logprobs=True,
        system_prompt="",
        context_docs=tuple(d.text for d in docs),
        tools_exposed=probe.tools_exposed,
    )
    return ctx.method.transform_request(req, ectx), ectx


def execute_units(
    units: Sequence[WorkUnit],
    probes: Sequence[Probe],
    ctx: ExecutionContext,
    checkpoint: Checkpoint | NullCheckpoint | None = None,
    verbose: bool = True,
) -> tuple[list[Generation], ExecutionStats]:
    """Run the planned units, resuming from the checkpoint."""
    ckpt = checkpoint or NullCheckpoint()
    by_id = {p.probe_id: p for p in probes}
    stats = ExecutionStats()
    out: list[Generation] = []

    backend = ctx.method.prepare(ctx.backend)
    containment_fp = ctx.method.fingerprint()
    backend_fp = backend.fingerprint()

    already = sum(1 for u in units if ckpt.is_done(u.unit_id))
    if verbose and already:
        print(f"resuming, {already} of {len(units)} units already recorded")

    # Every unit is walked on every run, including ones the checkpoint already recorded.
    # Resumability comes from the cache, where a hit is a lookup and no model call.
    # Skipping recorded units outright would leave their generations out of the returned
    # list, so scoring would quietly run on a subset of the probe suite. That is a much
    # worse failure than performing a few cheap lookups, because the resulting numbers
    # still look plausible.
    for n, unit in enumerate(units, start=1):
        probe = by_id.get(unit.probe_id)
        if probe is None:
            continue

        if ctx.dev_guard is not None:
            # tripwire: a method fitting on dev must not reach an eval entity
            ctx.dev_guard.require(unit.entity_id)

        gens = _run_unit(unit, probe, ctx, backend, backend_fp, containment_fp, stats)
        out.extend(gens)
        ckpt.mark(unit.unit_id)

        if verbose and (n % 50 == 0 or n == len(units)):
            print(
                f"  {n}/{len(units)} units, {stats.generated} generated, {stats.from_cache} cached"
            )

    if verbose and stats.deferred:
        print(
            f"deferred {len(stats.deferred)} units after repeated failures, retry on the next run"
        )

    return out, stats


def _run_unit(
    unit: WorkUnit,
    probe: Probe,
    ctx: ExecutionContext,
    backend: ModelBackend,
    backend_fp: str,
    containment_fp: str,
    stats: ExecutionStats,
) -> list[Generation]:
    requests: list[tuple[str, GenerationRequest, EntityContext]] = []
    found: list[Generation] = []

    for s in range(unit.k):
        req, ectx = build_request(ctx, probe, s)
        key = cache_key(backend_fp, containment_fp, req)
        hit = ctx.cache.get(key)
        if hit is not None:
            stats.from_cache += 1
            found.append(ctx.method.post_generate(hit, ectx))
        else:
            requests.append((key, req, ectx))

    if not requests:
        return found

    if ctx.tool_env is not None:
        ctx.tool_env.reset()

    try:
        raw = backend.generate([r for _, r, _ in requests])
    except BackendOOMError:
        stats.deferred.append(unit.unit_id)
        return found

    for (key, _req, ectx), gen in zip(requests, raw, strict=True):
        stamped = Generation.from_dict(
            {**gen.to_dict(), "cache_key": key, "containment_fp": containment_fp}
        )
        # write before use, so a kill after this line loses nothing
        ctx.cache.put(stamped)
        stats.generated += 1
        found.append(ctx.method.post_generate(stamped, ectx))

    return found


def group_by_probe(gens: Sequence[Generation]) -> dict[str, list[Generation]]:
    out: dict[str, list[Generation]] = {}
    for g in gens:
        out.setdefault(g.probe_id, []).append(g)
    for v in out.values():
        v.sort(key=lambda g: g.sample_index)
    return out
