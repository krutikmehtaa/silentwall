"""Stage orchestration.

Each stage reads from disk and writes to disk so a killed session costs one work unit
rather than the whole run. This module is what the CLI and the notebooks both call, so
there is exactly one implementation of the pipeline and the notebooks stay thin.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends.base import ModelBackend, make_backend
from .cache.keys import cache_key
from .cache.store import DiskCacheLayer, LayeredCache
from .config import SilentwallConfig, config_hash
from .containment.base import ContainmentMethod
from .containment.registry import build_method
from .containment.silentwall import SilentWallDefense
from .corpus.build import build_artifacts, build_corpus
from .corpus.splits import SplitAssignment, assign_splits
from .corpus.synth import synthesize_public_text
from .probes.generate import build_probe_suite
from .probes.memory import MemoryStore, prime_from_artifacts
from .probes.templates import behavioural_template_ids
from .probes.toolenv import DocStore, ToolEnv
from .report.build import build_audit_result, build_run_record
from .runner.checkpoint import Checkpoint
from .runner.execute import ExecutionContext, build_request, execute_units, group_by_probe
from .runner.plan import WorkUnit, check_budget, estimate, format_budget, plan_units
from .scoring.detector import fit_eval_detector
from .scoring.features import aggregate_entity_features, extract_features
from .scoring.leak import aggregate_leak, leak_curve, score_leak
from .scoring.stats import bootstrap_mean
from .types import (
    AuditResult,
    Corpus,
    DetectabilityResult,
    EntityFeatureVector,
    FeatureRow,
    Generation,
    Interval,
    LeakResult,
    LeakVerdict,
    PrivateArtifact,
    Probe,
    ProbeFamily,
    ProtectedField,
)

__all__ = ["Workspace", "prepare_workspace", "run_method", "run_sweep"]


@dataclass
class Workspace:
    """Everything stages 0 to 3 produce, shared across methods in a sweep."""

    cfg: SilentwallConfig
    corpus: Corpus
    artifacts: list[PrivateArtifact]
    probes: tuple[Probe, ...]
    splits: SplitAssignment
    fields: dict[str, ProtectedField]
    public_text: dict[str, str]

    @property
    def behavioural_templates(self) -> tuple[str, ...]:
        return behavioural_template_ids(self.cfg.behavioural_templates)


def prepare_workspace(cfg: SilentwallConfig, verbose: bool = True) -> Workspace:
    """Stages 0 through 2. CPU only, no model needed."""
    t0 = time.perf_counter()

    if verbose:
        mode = "offline" if cfg.corpus.synthetic else "live EDGAR"
        print(f"building corpus ({mode}), target {cfg.corpus.target_restricted} restricted")
    corpus = build_corpus(cfg)
    if verbose:
        print(
            f"corpus ready: {len(corpus.restricted)} restricted, {len(corpus.controls)} controls, "
            f"{len(corpus.pair_ids)} pairs, hash {corpus.manifest.corpus_hash[:12]}"
        )
        if corpus.exclusions:
            print(f"  excluded {len(corpus.exclusions)} filings, see the exclusion log")

    artifacts = build_artifacts(corpus, cfg.corpus.synthetic_seed)
    probes = build_probe_suite(
        corpus,
        template_pack_version=cfg.template_pack_version,
        n_behavioural=cfg.behavioural_templates,
        n_content_per_family=cfg.content_probes_per_family,
    )
    splits = assign_splits(corpus, cfg.split.split_seed, cfg.split.dev_fraction)

    fields = {f.field_id: f for d in corpus.deals for f in d.protected_fields}
    public_text = {
        e.entity_id: synthesize_public_text(e.display_name, e.sic_2digit, cfg.corpus.synthetic_seed)
        for e in corpus.entities
    }

    if verbose:
        n_content = sum(1 for p in probes if not p.is_behavioural)
        n_behav = sum(1 for p in probes if p.is_behavioural)
        print(f"probes: {n_content} content, {n_behav} behavioural, {len(probes)} total")
        print(
            f"split: {len(splits.dev.pair_ids)} dev pairs, {len(splits.eval.pair_ids)} eval pairs"
        )
        print(f"prepared in {time.perf_counter() - t0:.1f}s")

    return Workspace(cfg, corpus, artifacts, probes, splits, fields, public_text)


def _build_substrate(ws: Workspace, method: ContainmentMethod) -> tuple[ToolEnv, MemoryStore]:
    """Assemble the document store and memory.

    A method that declares requires_clean_substrate gets an environment with no private
    content at all. That is what makes the clean reference agent a genuine upper bound
    rather than a method with a strong filter.
    """
    clean = getattr(method, "requires_clean_substrate", False)
    store = DocStore()

    for entity_id, text in sorted(ws.public_text.items()):
        store.add(f"pub:{entity_id}", text, entity_id=entity_id, is_private=False)

    memory = MemoryStore()

    if not clean:
        deal_to_entity = {e.deal_id: e.entity_id for e in ws.corpus.entities if e.deal_id}
        for art in ws.artifacts:
            store.add(
                f"priv:{art.artifact_id}",
                art.text,
                entity_id=deal_to_entity.get(art.deal_id),
                is_private=True,
            )
        prime_from_artifacts(memory, ws.artifacts, deal_to_entity)

    return ToolEnv(store), memory


def _memorized_values(ws: Workspace) -> dict[str, list[str]]:
    """What the stub backend has absorbed into its weights.

    Simulates a model fine-tuned on the private corpus. Without this the parametric
    pathway would not exist and retrieval filtering would look perfect, which would be
    an artifact of the test double rather than a finding.
    """
    out: dict[str, list[str]] = {}
    for entity in ws.corpus.restricted:
        deal = ws.corpus.deal_for_entity(entity.entity_id)
        if deal is None:
            continue
        vals: list[str] = []
        for name in ("offer_price_per_share", "premium_pct"):
            f = deal.field_by_name(name)
            if f:
                vals.append(f.value_raw)
        if vals:
            out[entity.entity_id] = vals
    return out


def _make_backend(ws: Workspace, method: ContainmentMethod) -> ModelBackend:
    """Build the backend for a method.

    A method declaring requires_clean_substrate gets a model with no parametric
    knowledge of the protected fields, not merely a filtered document store. That
    distinction is the whole point of the clean reference: it stands for an agent that
    never encountered the private corpus at any stage, so withholding the documents
    while leaving the weights informed would make it a filtered agent rather than a
    clean one, and it would stop being a valid upper bound.
    """
    if ws.cfg.tier == "stub":
        clean = getattr(method, "requires_clean_substrate", False)
        return make_backend("stub", memorized={} if clean else _memorized_values(ws))
    return make_backend(ws.cfg.tier, model_id=ws.cfg.model_id)


def _cache(cfg: SilentwallConfig) -> LayeredCache:
    layers = [
        DiskCacheLayer(p, writable=(i == len(cfg.cache_layers) - 1))
        for i, p in enumerate(cfg.cache_layers)
    ]
    return LayeredCache(layers)


def _count_cached(
    ws: Workspace,
    ctx: ExecutionContext,
    units: Sequence[WorkUnit],
    backend_fp: str,
    containment_fp: str,
) -> int:
    """How many of the planned generations are already on disk.

    Costs nothing beyond building the prompts, which is what the runner would do
    anyway, and it makes the printed projection reflect reality.
    """
    probe_by_id = {p.probe_id: p for p in ws.probes}
    keys: list[str] = []
    for unit in units:
        probe = probe_by_id.get(unit.probe_id)
        if probe is None:
            continue
        for s in range(unit.k):
            req, _ = build_request(ctx, probe, s)
            keys.append(cache_key(backend_fp, containment_fp, req))
    return len(ctx.cache.contains_many(keys))


def _calibrate_on_dev(
    ws: Workspace,
    method: SilentWallDefense,
    backend: ModelBackend,
    cache: LayeredCache,
    verbose: bool = True,
) -> dict[str, float]:
    """Measure control-entity behaviour on the dev split and calibrate to match.

    Control entities only, dev split only. Controls hold no protected information, so
    nothing measured here is information the public side was not entitled to, and the
    dev restriction is what keeps the reported evaluation honest.

    The dev_guard on the execution context is the tripwire: if this pass ever reaches
    an eval entity it raises rather than quietly contaminating the calibration.
    """
    dev_controls = {e.entity_id for e in ws.splits.dev.entities if e.entity_class == "control"}
    if not dev_controls:
        return {}

    probes = [p for p in ws.probes if p.is_behavioural and p.entity_id in dev_controls]
    if not probes:
        return {}

    tool_env, memory = _build_substrate(ws, method)
    ctx = ExecutionContext(
        corpus=ws.corpus,
        cfg=ws.cfg,
        backend=backend,
        method=build_method("none"),  # measure the substrate, not our own effect
        cache=cache,
        tool_env=tool_env,
        memory=memory,
        public_text=ws.public_text,
        dev_guard=ws.splits.dev,
    )

    units = plan_units(
        probes,
        ws.cfg,
        config_hash(ws.cfg),
        "calibration-probe",
        backend.fingerprint(),
    )
    gens, _ = execute_units(units, probes, ctx, None, verbose=False)
    return method.calibrate_from_controls([g.text for g in gens])


def _leak_results(verdicts: Sequence[LeakVerdict], cfg: SilentwallConfig) -> list[LeakResult]:
    """Per-family leak@1 and leak@k with cluster bootstrap intervals over entities."""
    out: list[LeakResult] = []
    by_family: dict[ProbeFamily, list[LeakVerdict]] = {}
    for v in verdicts:
        by_family.setdefault(v.family, []).append(v)

    for family, vs in sorted(by_family.items(), key=lambda kv: kv[0].value):
        k = min(cfg.sampling.k, min(v.n_samples for v in vs))

        by_entity: dict[str, list[LeakVerdict]] = {}
        for v in vs:
            by_entity.setdefault(v.entity_id, []).append(v)

        c1 = [[float(aggregate_leak([v], 1)) for v in group] for group in by_entity.values()]
        ck = [[float(aggregate_leak([v], k)) for v in group] for group in by_entity.values()]

        curve: dict[int, float] = {}
        for kk in cfg.stats.leak_curve_k:
            if kk <= k:
                vals = [leak_curve(v, [kk]).get(kk, 0.0) for v in vs]
                curve[kk] = float(sum(vals) / len(vals)) if vals else 0.0

        out.append(
            LeakResult(
                family=family,
                leak_at_1=bootstrap_mean(
                    c1, cfg.stats.bootstrap_resamples // 4, cfg.stats.ci_level
                ),
                leak_at_k=bootstrap_mean(
                    ck, cfg.stats.bootstrap_resamples // 4, cfg.stats.ci_level
                ),
                k=k,
                n_probes=len(vs),
                n_entities=len(by_entity),
            )
        )
    return out


def _feature_vectors(
    ws: Workspace, gens: Sequence[Generation], eval_only: bool = True
) -> list[EntityFeatureVector]:
    """Behavioural feature vectors, one per entity."""
    by_probe = group_by_probe(gens)
    probe_by_id = {p.probe_id: p for p in ws.probes}

    rows_by_entity: dict[str, list[FeatureRow]] = {}
    for probe_id, group in by_probe.items():
        probe = probe_by_id.get(probe_id)
        if probe is None or not probe.is_behavioural:
            continue
        entity = ws.corpus.entity(probe.entity_id)
        row = extract_features(group, probe, entity)
        rows_by_entity.setdefault(entity.entity_id, []).append(row)

    vectors: list[EntityFeatureVector] = []
    for entity_id, rows in rows_by_entity.items():
        try:
            split = ws.splits.split_of(entity_id)
        except KeyError:
            continue
        if eval_only and split != "eval":
            continue
        vec = aggregate_entity_features(rows, ws.behavioural_templates, split)
        if vec is not None:
            vectors.append(vec)
    return vectors


def _utility(
    ws: Workspace, verdicts: Sequence[LeakVerdict], cfg: SilentwallConfig
) -> dict[str, Interval]:
    """Retention on the private side and collateral damage on controls.

    Private-side retention uses the direct family as a proxy for whether the entitled
    audience can still reach the fact. Control-entity utility is the over-forgetting
    probe: if it moves, containment is bleeding past its target.
    """
    out: dict[str, Interval] = {}

    direct = [v for v in verdicts if v.family is ProbeFamily.DIRECT]
    if direct:
        by_entity: dict[str, list[float]] = {}
        for v in direct:
            by_entity.setdefault(v.entity_id, []).append(
                float(aggregate_leak([v], min(cfg.sampling.k, v.n_samples)))
            )
        out["private_side_retention"] = bootstrap_mean(
            list(by_entity.values()), cfg.stats.bootstrap_resamples // 4, cfg.stats.ci_level
        )
    return out


def run_method(
    ws: Workspace,
    method_id: str,
    verbose: bool = True,
    confirm_budget: bool = False,
    method_params: dict[str, Any] | None = None,
) -> AuditResult:
    """Stages 3 through 6 for one containment method."""
    cfg = ws.cfg
    t0 = time.perf_counter()

    if verbose:
        print(f"\nmethod: {method_id}")

    method = build_method(method_id, **(method_params or {}))
    tool_env, memory = _build_substrate(ws, method)

    # calibration and fitting happen on the dev split only
    method.fit(ws.splits.dev, ws.artifacts)

    backend = _make_backend(ws, method)
    cache = _cache(cfg)

    if isinstance(method, SilentWallDefense):
        for entity_id, text in ws.public_text.items():
            method.set_public_text(entity_id, text)
        stats_c = _calibrate_on_dev(ws, method, backend, cache, verbose=verbose)
        if verbose and stats_c:
            print(
                f"  calibrated on dev controls: target length {stats_c['target_length']:.1f} "
                f"tokens, hedge rate {stats_c['hedge_rate']:.2f}"
            )

    ctx = ExecutionContext(
        corpus=ws.corpus,
        cfg=cfg,
        backend=backend,
        method=method,
        cache=cache,
        tool_env=tool_env,
        memory=memory,
        public_text=ws.public_text,
        dev_guard=None,
    )

    cfg_hash = config_hash(cfg)
    prepared = method.prepare(backend)
    units = plan_units(ws.probes, cfg, cfg_hash, method.fingerprint(), prepared.fingerprint())

    # Count real cache hits before deciding whether the run is affordable. Reporting
    # zero here would make an almost fully cached run look like a full-price one, which
    # defeats the point of printing a budget at all.
    n_cached = _count_cached(ws, ctx, units, prepared.fingerprint(), method.fingerprint())
    est = estimate(units, n_cached, cfg.tier)
    if verbose:
        print(format_budget(est))
    check_budget(est, cfg, confirmed=confirm_budget)

    ckpt = Checkpoint(cfg.logs_dir / f"checkpoint_{method_id}_{cfg_hash[:8]}.txt")
    gens, stats = execute_units(units, ws.probes, ctx, ckpt, verbose=verbose)

    # scoring
    by_probe = group_by_probe(gens)
    probe_by_id = {p.probe_id: p for p in ws.probes}

    verdicts: list[LeakVerdict] = []
    for probe_id, group in by_probe.items():
        probe = probe_by_id.get(probe_id)
        if probe is None or probe.is_behavioural:
            continue
        verdicts.extend(score_leak(probe, group, ws.fields))

    leak = _leak_results(verdicts, cfg)
    vectors = _feature_vectors(ws, gens, eval_only=True)

    detectability: list[DetectabilityResult] = []
    if len({v.entity_class for v in vectors}) == 2:
        detectability.append(fit_eval_detector(vectors, cfg.stats, "logreg_primary", seed=17))
        if len(vectors) >= 20:
            detectability.append(fit_eval_detector(vectors, cfg.stats, "gbt_secondary", seed=17))
    elif verbose:
        print("  not enough class variety in the eval split to fit a detector")

    utility = _utility(ws, verdicts, cfg)

    run = build_run_record(
        cfg=cfg,
        corpus_hash=ws.corpus.manifest.corpus_hash,
        backend_fp=prepared.fingerprint(),
        containment_fp=method.fingerprint(),
        split_audit=ws.splits.audit(),
        elapsed_seconds=time.perf_counter() - t0,
        generations_total=stats.total,
        generations_from_cache=stats.from_cache,
        quarantined=cache.quarantined,
        deferred=len(stats.deferred),
    )

    result = build_audit_result(cfg, method_id, run, leak, detectability, utility, gens)

    if verbose:
        det = result.primary_detectability
        worst = max((lr.leak_at_k.point for lr in leak), default=0.0)
        auc_txt = f"{det.auc.point:.3f}" if det else "n/a"
        print(
            f"  worst-family leak@k {worst:.3f}, detectability AUC {auc_txt}, "
            f"{stats.total} generations in {run.elapsed_seconds:.1f}s"
        )

    return result


def run_sweep(
    cfg: SilentwallConfig,
    verbose: bool = True,
    confirm_budget: bool = False,
) -> list[AuditResult]:
    """Full sweep across every configured method on one shared workspace."""
    ws = prepare_workspace(cfg, verbose=verbose)
    results: list[AuditResult] = []
    for method_id in cfg.methods:
        params = dict(cfg.method_params.get(method_id, {}))
        results.append(
            run_method(
                ws,
                method_id,
                verbose=verbose,
                confirm_budget=confirm_budget,
                method_params=params,
            )
        )
    return results


def save_workspace(ws: Workspace, out_dir: Path | str) -> Path:
    """Persist the corpus and probe suite so later stages can run standalone."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "manifest.json").write_text(
        json.dumps(ws.corpus.manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_jsonl(out / "deals.jsonl", [d.to_dict() for d in ws.corpus.deals])
    _write_jsonl(out / "entities.jsonl", [e.to_dict() for e in ws.corpus.entities])
    _write_jsonl(out / "exclusions.jsonl", [x.to_dict() for x in ws.corpus.exclusions])
    _write_jsonl(out / "artifacts.jsonl", [a.to_dict() for a in ws.artifacts])
    _write_jsonl(out / "probes.jsonl", [p.to_dict() for p in ws.probes])
    (out / "split_audit.json").write_text(
        json.dumps(ws.splits.audit(), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return out


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
