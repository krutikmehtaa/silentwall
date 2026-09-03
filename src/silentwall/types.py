"""Data models.

Every record is a frozen dataclass so it can be hashed, cached and compared by
value. Serialization goes through to_dict and from_dict rather than asdict, because
we need dates as ISO strings and tuples restored as tuples, not lists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Literal

from .hashing import hash_obj

__all__ = [
    "ProbeFamily",
    "EntityClass",
    "SplitName",
    "FieldName",
    "MatchMode",
    "ProtectedField",
    "DealRecord",
    "Entity",
    "ExclusionRecord",
    "CorpusManifest",
    "Corpus",
    "PrivateArtifact",
    "Probe",
    "TokenTrace",
    "Generation",
    "LeakVerdict",
    "FeatureRow",
    "EntityFeatureVector",
    "Interval",
    "DetectabilityResult",
    "LeakResult",
    "RunRecord",
    "AuditResult",
    "CONTENT_FAMILIES",
]


class ProbeFamily(str, Enum):
    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    INFERENCE = "inference_chain"
    TOOL = "tool_mediated"
    MEMORY = "memory_retrieval"
    BEHAVIOURAL = "behavioural"


#: Families that target a protected field. Behavioural is the odd one out: it
#: never references protected content, which is what makes it a detectability
#: probe rather than a leakage probe.
CONTENT_FAMILIES: tuple[ProbeFamily, ...] = (
    ProbeFamily.DIRECT,
    ProbeFamily.PARAPHRASE,
    ProbeFamily.INFERENCE,
    ProbeFamily.TOOL,
    ProbeFamily.MEMORY,
)

EntityClass = Literal["restricted", "control"]
SplitName = Literal["dev", "eval"]
MatchMode = Literal["exact", "numeric_normalized"]
FieldName = Literal[
    "offer_price_per_share",
    "announcement_date",
    "premium_pct",
    "consideration_type",
    "expected_close",
]
ExclusionReason = Literal[
    "no_item_101",
    "no_target",
    "no_acquirer",
    "no_numeric_field",
    "ambiguous_parties",
    "fetch_failed",
    "duplicate_deal",
]
ArtifactKind = Literal[
    "engagement_memo",
    "fairness_opinion",
    "valuation_summary",
    "diligence_notes",
]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if value is not None else ()


@dataclass(frozen=True, slots=True)
class ProtectedField:
    """A single fact that must be contained."""

    field_id: str
    name: FieldName
    value_raw: str
    value_normalized: str
    source_accession: str
    source_span: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "name": self.name,
            "value_raw": self.value_raw,
            "value_normalized": self.value_normalized,
            "source_accession": self.source_accession,
            "source_span": list(self.source_span),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ProtectedField:
        return cls(
            field_id=d["field_id"],
            name=d["name"],
            value_raw=d["value_raw"],
            value_normalized=d["value_normalized"],
            source_accession=d["source_accession"],
            source_span=(int(d["source_span"][0]), int(d["source_span"][1])),
        )


@dataclass(frozen=True, slots=True)
class DealRecord:
    """One M&A event extracted from public filings."""

    deal_id: str
    acquirer_name: str
    target_name: str
    target_cik: str
    announcement_date: date
    sic_2digit: str
    size_metric_value: float
    size_band: int
    protected_fields: tuple[ProtectedField, ...]
    source_accessions: tuple[str, ...]
    acquirer_cik: str | None = None

    def field_by_name(self, name: FieldName) -> ProtectedField | None:
        for f in self.protected_fields:
            if f.name == name:
                return f
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "acquirer_name": self.acquirer_name,
            "acquirer_cik": self.acquirer_cik,
            "target_name": self.target_name,
            "target_cik": self.target_cik,
            "announcement_date": self.announcement_date.isoformat(),
            "sic_2digit": self.sic_2digit,
            "size_metric_value": self.size_metric_value,
            "size_band": self.size_band,
            "protected_fields": [f.to_dict() for f in self.protected_fields],
            "source_accessions": list(self.source_accessions),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DealRecord:
        return cls(
            deal_id=d["deal_id"],
            acquirer_name=d["acquirer_name"],
            acquirer_cik=d.get("acquirer_cik"),
            target_name=d["target_name"],
            target_cik=d["target_cik"],
            announcement_date=date.fromisoformat(d["announcement_date"]),
            sic_2digit=d["sic_2digit"],
            size_metric_value=float(d["size_metric_value"]),
            size_band=int(d["size_band"]),
            protected_fields=tuple(ProtectedField.from_dict(f) for f in d["protected_fields"]),
            source_accessions=_as_tuple(d["source_accessions"]),
        )


@dataclass(frozen=True, slots=True)
class Entity:
    """A company in the study, restricted or control.

    pair_id is shared by a restricted entity and the control it was matched to.
    Splits partition on pair_id, never on entity_id, so a matched pair never
    straddles a fold boundary.
    """

    entity_id: str
    display_name: str
    cik: str
    entity_class: EntityClass
    sic_2digit: str
    size_band: int
    pair_id: str
    deal_id: str | None = None

    @property
    def is_restricted(self) -> bool:
        return self.entity_class == "restricted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "display_name": self.display_name,
            "cik": self.cik,
            "entity_class": self.entity_class,
            "sic_2digit": self.sic_2digit,
            "size_band": self.size_band,
            "pair_id": self.pair_id,
            "deal_id": self.deal_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Entity:
        return cls(
            entity_id=d["entity_id"],
            display_name=d["display_name"],
            cik=d["cik"],
            entity_class=d["entity_class"],
            sic_2digit=d["sic_2digit"],
            size_band=int(d["size_band"]),
            pair_id=d["pair_id"],
            deal_id=d.get("deal_id"),
        )


@dataclass(frozen=True, slots=True)
class ExclusionRecord:
    """Why a candidate filing did not become a DealRecord."""

    accession: str
    reason: ExclusionReason
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"accession": self.accession, "reason": self.reason, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExclusionRecord:
        return cls(accession=d["accession"], reason=d["reason"], detail=d.get("detail", ""))


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    corpus_hash: str
    n_restricted: int
    n_control: int
    n_excluded: int
    edgar_query: Mapping[str, Any]
    template_pack_version: str
    built_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_hash": self.corpus_hash,
            "n_restricted": self.n_restricted,
            "n_control": self.n_control,
            "n_excluded": self.n_excluded,
            "edgar_query": dict(self.edgar_query),
            "template_pack_version": self.template_pack_version,
            "built_at_utc": self.built_at_utc,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CorpusManifest:
        return cls(
            corpus_hash=d["corpus_hash"],
            n_restricted=int(d["n_restricted"]),
            n_control=int(d["n_control"]),
            n_excluded=int(d["n_excluded"]),
            edgar_query=dict(d["edgar_query"]),
            template_pack_version=d["template_pack_version"],
            built_at_utc=d["built_at_utc"],
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    manifest: CorpusManifest
    deals: tuple[DealRecord, ...]
    entities: tuple[Entity, ...]
    exclusions: tuple[ExclusionRecord, ...] = ()

    @property
    def restricted(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.entity_class == "restricted")

    @property
    def controls(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.entity_class == "control")

    def deal(self, deal_id: str) -> DealRecord:
        for d in self.deals:
            if d.deal_id == deal_id:
                return d
        raise KeyError(deal_id)

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        raise KeyError(entity_id)

    def deal_for_entity(self, entity_id: str) -> DealRecord | None:
        ent = self.entity(entity_id)
        return self.deal(ent.deal_id) if ent.deal_id else None

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(sorted({e.pair_id for e in self.entities}))


@dataclass(frozen=True, slots=True)
class PrivateArtifact:
    """A document the deal team would hold. Synthesized, never real."""

    artifact_id: str
    deal_id: str
    kind: ArtifactKind
    text: str
    embeds_field_ids: tuple[str, ...]
    template_id: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "deal_id": self.deal_id,
            "kind": self.kind,
            "text": self.text,
            "embeds_field_ids": list(self.embeds_field_ids),
            "template_id": self.template_id,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PrivateArtifact:
        return cls(
            artifact_id=d["artifact_id"],
            deal_id=d["deal_id"],
            kind=d["kind"],
            text=d["text"],
            embeds_field_ids=_as_tuple(d["embeds_field_ids"]),
            template_id=d["template_id"],
            seed=int(d["seed"]),
        )


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    family: ProbeFamily
    entity_id: str
    template_id: str
    prompt: str
    target_field_ids: tuple[str, ...] = ()
    tools_exposed: tuple[str, ...] = ()
    memory_primed: bool = False

    @property
    def is_behavioural(self) -> bool:
        return self.family is ProbeFamily.BEHAVIOURAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "family": self.family.value,
            "entity_id": self.entity_id,
            "template_id": self.template_id,
            "prompt": self.prompt,
            "target_field_ids": list(self.target_field_ids),
            "tools_exposed": list(self.tools_exposed),
            "memory_primed": self.memory_primed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Probe:
        return cls(
            probe_id=d["probe_id"],
            family=ProbeFamily(d["family"]),
            entity_id=d["entity_id"],
            template_id=d["template_id"],
            prompt=d["prompt"],
            target_field_ids=_as_tuple(d["target_field_ids"]),
            tools_exposed=_as_tuple(d["tools_exposed"]),
            memory_primed=bool(d.get("memory_primed", False)),
        )


@dataclass(frozen=True, slots=True)
class TokenTrace:
    """Per-token decoding detail, when the backend can supply it."""

    token_ids: tuple[int, ...]
    chosen_logprobs: tuple[float, ...] | None = None
    topm_logprobs: tuple[tuple[float, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "chosen_logprobs": list(self.chosen_logprobs) if self.chosen_logprobs else None,
            "topm_logprobs": [list(r) for r in self.topm_logprobs] if self.topm_logprobs else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TokenTrace:
        topm = d.get("topm_logprobs")
        chosen = d.get("chosen_logprobs")
        return cls(
            token_ids=_as_tuple(d["token_ids"]),
            chosen_logprobs=tuple(float(x) for x in chosen) if chosen else None,
            topm_logprobs=tuple(tuple(float(x) for x in r) for r in topm) if topm else None,
        )


@dataclass(frozen=True, slots=True)
class Generation:
    """One sampled completion. The unit that gets cached."""

    cache_key: str
    probe_id: str
    entity_id: str
    sample_index: int
    seed: int
    text: str
    n_tokens: int
    backend_fp: str
    containment_fp: str
    trace: TokenTrace | None = None
    tool_calls: tuple[str, ...] = ()
    latency_ms: float = 0.0
    latency_trustworthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "probe_id": self.probe_id,
            "entity_id": self.entity_id,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "backend_fp": self.backend_fp,
            "containment_fp": self.containment_fp,
            "trace": self.trace.to_dict() if self.trace else None,
            "tool_calls": list(self.tool_calls),
            "latency_ms": self.latency_ms,
            "latency_trustworthy": self.latency_trustworthy,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Generation:
        tr = d.get("trace")
        return cls(
            cache_key=d["cache_key"],
            probe_id=d["probe_id"],
            entity_id=d["entity_id"],
            sample_index=int(d["sample_index"]),
            seed=int(d["seed"]),
            text=d["text"],
            n_tokens=int(d["n_tokens"]),
            backend_fp=d["backend_fp"],
            containment_fp=d["containment_fp"],
            trace=TokenTrace.from_dict(tr) if tr else None,
            tool_calls=_as_tuple(d.get("tool_calls")),
            latency_ms=float(d.get("latency_ms", 0.0)),
            latency_trustworthy=bool(d.get("latency_trustworthy", True)),
        )


@dataclass(frozen=True, slots=True)
class LeakVerdict:
    """Whether a protected field surfaced, recorded per sample.

    Storing the whole vector rather than a collapsed boolean is what lets leak@1,
    leak@4 and leak@16 all come out of one sampling budget.
    """

    probe_id: str
    entity_id: str
    family: ProbeFamily
    field_id: str
    per_sample_hit: tuple[bool, ...]
    match_mode: MatchMode = "numeric_normalized"

    @property
    def n_samples(self) -> int:
        return len(self.per_sample_hit)

    @property
    def n_hits(self) -> int:
        return sum(self.per_sample_hit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "entity_id": self.entity_id,
            "family": self.family.value,
            "field_id": self.field_id,
            "per_sample_hit": list(self.per_sample_hit),
            "match_mode": self.match_mode,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LeakVerdict:
        return cls(
            probe_id=d["probe_id"],
            entity_id=d["entity_id"],
            family=ProbeFamily(d["family"]),
            field_id=d["field_id"],
            per_sample_hit=tuple(bool(x) for x in d["per_sample_hit"]),
            match_mode=d.get("match_mode", "numeric_normalized"),
        )


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Behavioural features for one (entity, template) pair."""

    entity_id: str
    pair_id: str
    template_id: str
    entity_class: EntityClass
    values: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "pair_id": self.pair_id,
            "template_id": self.template_id,
            "entity_class": self.entity_class,
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FeatureRow:
        return cls(
            entity_id=d["entity_id"],
            pair_id=d["pair_id"],
            template_id=d["template_id"],
            entity_class=d["entity_class"],
            values=dict(d["values"]),
        )


@dataclass(frozen=True, slots=True)
class EntityFeatureVector:
    """Behavioural features aggregated to one vector per entity."""

    entity_id: str
    pair_id: str
    entity_class: EntityClass
    split: SplitName
    primary: Mapping[str, float]
    secondary: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return 1 if self.entity_class == "restricted" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "pair_id": self.pair_id,
            "entity_class": self.entity_class,
            "split": self.split,
            "primary": dict(self.primary),
            "secondary": dict(self.secondary),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EntityFeatureVector:
        return cls(
            entity_id=d["entity_id"],
            pair_id=d["pair_id"],
            entity_class=d["entity_class"],
            split=d["split"],
            primary=dict(d["primary"]),
            secondary=dict(d.get("secondary", {})),
        )


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    lo: float
    hi: float
    level: float = 0.95
    method: str = "cluster_bootstrap"

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "level": self.level,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Interval:
        return cls(
            point=float(d["point"]),
            lo=float(d["lo"]),
            hi=float(d["hi"]),
            level=float(d.get("level", 0.95)),
            method=d.get("method", "cluster_bootstrap"),
        )

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


@dataclass(frozen=True, slots=True)
class DetectabilityResult:
    """The headline. Can an observer locate the wall from behaviour alone?"""

    auc: Interval
    detector_id: str
    n_pairs: int
    auc_by_repeat: tuple[float, ...] = ()
    permutation_p: float = float("nan")
    permutation_p_adjusted: float = float("nan")
    transfer_auc: Interval | None = None
    feature_importance: Mapping[str, Interval] = field(default_factory=dict)
    undetectable_claim: bool = False
    power_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc.to_dict(),
            "detector_id": self.detector_id,
            "n_pairs": self.n_pairs,
            "auc_by_repeat": list(self.auc_by_repeat),
            "permutation_p": self.permutation_p,
            "permutation_p_adjusted": self.permutation_p_adjusted,
            "transfer_auc": self.transfer_auc.to_dict() if self.transfer_auc else None,
            "feature_importance": {k: v.to_dict() for k, v in self.feature_importance.items()},
            "undetectable_claim": self.undetectable_claim,
            "power_note": self.power_note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DetectabilityResult:
        ta = d.get("transfer_auc")
        return cls(
            auc=Interval.from_dict(d["auc"]),
            detector_id=d["detector_id"],
            n_pairs=int(d["n_pairs"]),
            auc_by_repeat=tuple(float(x) for x in d.get("auc_by_repeat", ())),
            permutation_p=float(d.get("permutation_p", float("nan"))),
            permutation_p_adjusted=float(d.get("permutation_p_adjusted", float("nan"))),
            transfer_auc=Interval.from_dict(ta) if ta else None,
            feature_importance={
                k: Interval.from_dict(v) for k, v in d.get("feature_importance", {}).items()
            },
            undetectable_claim=bool(d.get("undetectable_claim", False)),
            power_note=d.get("power_note", ""),
        )


@dataclass(frozen=True, slots=True)
class LeakResult:
    family: ProbeFamily
    leak_at_1: Interval
    leak_at_k: Interval
    k: int
    n_probes: int
    n_entities: int
    curve: Mapping[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "leak_at_1": self.leak_at_1.to_dict(),
            "leak_at_k": self.leak_at_k.to_dict(),
            "k": self.k,
            "n_probes": self.n_probes,
            "n_entities": self.n_entities,
            "curve": {str(k): v for k, v in self.curve.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LeakResult:
        return cls(
            family=ProbeFamily(d["family"]),
            leak_at_1=Interval.from_dict(d["leak_at_1"]),
            leak_at_k=Interval.from_dict(d["leak_at_k"]),
            k=int(d["k"]),
            n_probes=int(d["n_probes"]),
            n_entities=int(d["n_entities"]),
            curve={int(k): float(v) for k, v in d.get("curve", {}).items()},
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Everything needed to reproduce a run, including the split audit."""

    run_id: str
    config_hash: str
    corpus_hash: str
    backend_fp: str
    containment_fp: str
    library_versions: Mapping[str, str]
    seeds: Mapping[str, int]
    dev_ids_hash: str
    eval_ids_hash: str
    splits_disjoint: bool
    elapsed_seconds: float = 0.0
    generations_total: int = 0
    generations_from_cache: int = 0
    quarantined: int = 0
    deferred: int = 0
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "corpus_hash": self.corpus_hash,
            "backend_fp": self.backend_fp,
            "containment_fp": self.containment_fp,
            "library_versions": dict(self.library_versions),
            "seeds": dict(self.seeds),
            "dev_ids_hash": self.dev_ids_hash,
            "eval_ids_hash": self.eval_ids_hash,
            "splits_disjoint": self.splits_disjoint,
            "elapsed_seconds": self.elapsed_seconds,
            "generations_total": self.generations_total,
            "generations_from_cache": self.generations_from_cache,
            "quarantined": self.quarantined,
            "deferred": self.deferred,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunRecord:
        return cls(
            run_id=d["run_id"],
            config_hash=d["config_hash"],
            corpus_hash=d["corpus_hash"],
            backend_fp=d["backend_fp"],
            containment_fp=d["containment_fp"],
            library_versions=dict(d["library_versions"]),
            seeds={k: int(v) for k, v in d["seeds"].items()},
            dev_ids_hash=d["dev_ids_hash"],
            eval_ids_hash=d["eval_ids_hash"],
            splits_disjoint=bool(d["splits_disjoint"]),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            generations_total=int(d.get("generations_total", 0)),
            generations_from_cache=int(d.get("generations_from_cache", 0)),
            quarantined=int(d.get("quarantined", 0)),
            deferred=int(d.get("deferred", 0)),
            git_commit=d.get("git_commit"),
        )


@dataclass(frozen=True, slots=True)
class AuditResult:
    run: RunRecord
    method_id: str
    leak: tuple[LeakResult, ...]
    detectability: tuple[DetectabilityResult, ...]
    utility: Mapping[str, Interval] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @property
    def primary_detectability(self) -> DetectabilityResult | None:
        for d in self.detectability:
            if d.detector_id == "logreg_primary":
                return d
        return self.detectability[0] if self.detectability else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "method_id": self.method_id,
            "leak": [x.to_dict() for x in self.leak],
            "detectability": [x.to_dict() for x in self.detectability],
            "utility": {k: v.to_dict() for k, v in self.utility.items()},
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AuditResult:
        return cls(
            run=RunRecord.from_dict(d["run"]),
            method_id=d["method_id"],
            leak=tuple(LeakResult.from_dict(x) for x in d["leak"]),
            detectability=tuple(DetectabilityResult.from_dict(x) for x in d["detectability"]),
            utility={k: Interval.from_dict(v) for k, v in d.get("utility", {}).items()},
            limitations=_as_tuple(d.get("limitations")),
        )


def make_probe_id(
    family: ProbeFamily,
    template_id: str,
    template_pack_version: str,
    entity_id: str,
    slots: Mapping[str, Any],
) -> str:
    """Stable probe identity. Same inputs give the same id across sessions."""
    return hash_obj(family.value, template_id, template_pack_version, entity_id, slots)[:16]
