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
from dataclasses import dataclass, field, replace

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

    # One code path for every backend, including the stub. An earlier version kept a
    # separate per-unit loop for the stub and used the batched path only on GPU, which
    # meant the path that actually runs the expensive work was the one no test touched.
    # The stub seeds each generation from the request rather than the batch, so batching
    # does not change its output and determinism is preserved.
    return _execute_batched(
        units, by_id, ctx, backend, backend_fp, containment_fp, ckpt, stats, out, verbose
    )


def _execute_batched(
    units: Sequence[WorkUnit],
    by_id: dict[str, Probe],
    ctx: ExecutionContext,
    backend: ModelBackend,
    backend_fp: str,
    containment_fp: str,
    ckpt: Checkpoint | NullCheckpoint,
    stats: ExecutionStats,
    out: list[Generation],
    verbose: bool,
) -> tuple[list[Generation], ExecutionStats]:
    """Dispatch generations batched across probes rather than one probe at a time.

    Autoregressive decode on a memory-bound accelerator costs the same per step whether
    the batch holds 8 sequences or 32, because every step reads the full weight matrix
    either way. A T4 at 320 GB/s spends about 10ms per step on a 3GB fp16 model no
    matter what. Sending one probe at a time therefore paid full price for k sequences,
    where k is 8 in the iterate profile, and left three quarters of the batch empty.

    Requests are also sorted by prompt length before batching, so padding waste stays
    small. Note the tradeoff: batch composition can perturb fp16 results very slightly
    through padding and RNG state, so exact per-sample reproducibility on GPU tiers is
    not guaranteed the way it is on the stub. Cache keys do not encode batch membership,
    which is the standard compromise and is recorded in the report limitations.
    """
    # One retrieval pass per probe rather than per sample. Retrieval is identical across
    # the k samples of a probe, so doing it k times was pure waste.
    pending: list[tuple[str, GenerationRequest, EntityContext, str]] = []
    by_unit: dict[str, list[Generation]] = {}

    for unit in units:
        probe = by_id.get(unit.probe_id)
        if probe is None:
            continue
        if ctx.dev_guard is not None:
            ctx.dev_guard.require(unit.entity_id)

        base_req, ectx = build_request(ctx, probe, 0)
        for s in range(unit.k):
            req = base_req if s == 0 else replace(base_req, sample_index=s)
            key = cache_key(backend_fp, containment_fp, req)
            hit = ctx.cache.get(key)
            if hit is not None:
                stats.from_cache += 1
                by_unit.setdefault(unit.unit_id, []).append(ctx.method.post_generate(hit, ectx))
            else:
                pending.append((key, req, ectx, unit.unit_id))

    batch_size = max(1, int(getattr(backend, "batch_size", 8)))
    if verbose:
        print(
            f"  {len(pending)} generations to run, {stats.from_cache} from cache, "
            f"batch size {batch_size}"
        )

    # shortest prompts first, so each batch pads to a similar length
    pending.sort(key=lambda item: (len(item[1].full_prompt()), item[0]))

    if ctx.tool_env is not None:
        ctx.tool_env.reset()

    done = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        try:
            raw = backend.generate([req for _, req, _, _ in chunk])
        except BackendOOMError:
            for _, _, _, uid in chunk:
                if uid not in stats.deferred:
                    stats.deferred.append(uid)
            continue

        for (key, _req, ectx, uid), gen in zip(chunk, raw, strict=True):
            stamped = Generation.from_dict(
                {**gen.to_dict(), "cache_key": key, "containment_fp": containment_fp}
            )
            ctx.cache.put(stamped)
            stats.generated += 1
            by_unit.setdefault(uid, []).append(ctx.method.post_generate(stamped, ectx))

        done += len(chunk)
        if verbose and (done % (batch_size * 20) < batch_size or done >= len(pending)):
            print(f"  {done}/{len(pending)} generations, {stats.from_cache} cached")

    deferred = set(stats.deferred)
    for unit in units:
        gens = by_unit.get(unit.unit_id, [])
        out.extend(gens)
        if unit.unit_id not in deferred:
            ckpt.mark(unit.unit_id)

    if verbose and stats.deferred:
        print(f"deferred {len(stats.deferred)} units after repeated failures, retry next run")

    return out, stats


def group_by_probe(gens: Sequence[Generation]) -> dict[str, list[Generation]]:
    out: dict[str, list[Generation]] = {}
    for g in gens:
        out.setdefault(g.probe_id, []).append(g)
    for v in out.values():
        v.sort(key=lambda g: g.sample_index)
    return out
