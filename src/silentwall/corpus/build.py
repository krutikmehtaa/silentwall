"""Corpus construction, stage 0. CPU only.

Two routes to a corpus. The live route pulls real 8-K filings from EDGAR. The
offline route generates a structurally identical corpus from seeded templates, which
is what lets the smoke profile and CI run the full pipeline with no network and no
GPU. Both produce the same types, so nothing downstream knows which was used.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Sequence
from pathlib import Path

from ..config import CorpusConfig, SilentwallConfig
from ..errors import CorpusFetchError, MatchingInfeasibleError
from ..hashing import hash_obj, short_hash, stable_seed
from ..types import (
    Corpus,
    CorpusManifest,
    DealRecord,
    Entity,
    ExclusionRecord,
    PrivateArtifact,
    ProtectedField,
)
from .controls import Candidate, build_entities, match_controls
from .edgar import DiskHttpCache, EdgarClient, IndexRow
from .parse import parse_or_exclude
from .synth import TEMPLATE_PACK_VERSION, synthesize_artifacts

__all__ = ["build_corpus", "build_synthetic_corpus", "build_live_corpus", "corpus_content_hash"]

_MA_FORM_HINTS = ("8-K", "S-4", "SC 13D", "SC 14D9", "DEFM14A")


def corpus_content_hash(
    deals: Sequence[DealRecord], entities: Sequence[Entity], template_pack: str
) -> str:
    """Hash over content, insensitive to record order.

    Sorting before hashing is the whole point: two builds that found the same deals
    in a different order are the same corpus and must produce the same hash.
    """
    return hash_obj(
        [d.to_dict() for d in sorted(deals, key=lambda d: d.deal_id)],
        [e.to_dict() for e in sorted(entities, key=lambda e: e.entity_id)],
        template_pack,
    )


def _assemble(
    deals: Sequence[DealRecord],
    entities: Sequence[Entity],
    exclusions: Sequence[ExclusionRecord],
    edgar_query: dict[str, object],
) -> Corpus:
    n_restricted = sum(1 for e in entities if e.entity_class == "restricted")
    n_control = sum(1 for e in entities if e.entity_class == "control")
    manifest = CorpusManifest(
        corpus_hash=corpus_content_hash(deals, entities, TEMPLATE_PACK_VERSION),
        n_restricted=n_restricted,
        n_control=n_control,
        n_excluded=len(exclusions),
        edgar_query=edgar_query,
        template_pack_version=TEMPLATE_PACK_VERSION,
        built_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    return Corpus(
        manifest=manifest,
        deals=tuple(sorted(deals, key=lambda d: d.deal_id)),
        entities=tuple(sorted(entities, key=lambda e: e.entity_id)),
        exclusions=tuple(exclusions),
    )


# Offline corpus


_ACQ_PREFIX = (
    "Northbridge",
    "Calder",
    "Ashford",
    "Brightwater",
    "Kestrel",
    "Halcyon",
    "Meridian",
    "Sandpiper",
    "Thornhill",
    "Westgate",
    "Ironwood",
    "Larkspur",
)
_TGT_PREFIX = (
    "Verity",
    "Coastline",
    "Pinnacle",
    "Redstone",
    "Silverline",
    "Tessera",
    "Orchard",
    "Bluepeak",
    "Granite",
    "Fairmount",
    "Lakeshore",
    "Cobalt",
)
_SUFFIX = ("Holdings", "Group", "Industries", "Technologies", "Partners", "Systems", "Labs")
_SECTORS = ("20", "28", "35", "36", "38", "48", "49", "51", "60", "63", "67", "73", "80", "87")
_CONSIDERATION = ("all-cash", "cash and stock", "all-stock")
_QUARTERS = ("first quarter of 2020", "second quarter of 2020", "third quarter of 2020")


def _synth_name(rng: random.Random, prefixes: Sequence[str], used: set[str]) -> str:
    for _ in range(500):
        name = f"{rng.choice(prefixes)} {rng.choice(_SUFFIX)}"
        if name not in used:
            used.add(name)
            return name
    # deterministic fallback keeps this a total function
    name = f"{rng.choice(prefixes)} {rng.choice(_SUFFIX)} {len(used)}"
    used.add(name)
    return name


def build_synthetic_corpus(cfg: CorpusConfig) -> Corpus:
    """Structurally realistic corpus with no network access.

    Field values, sectors and dates vary the way real ones do, so control matching,
    size banding and leak scoring all get exercised. Only the text is invented.
    """
    n = cfg.target_restricted
    rng = random.Random(stable_seed(cfg.synthetic_seed, n, "synthetic-corpus-v1"))
    used_names: set[str] = set()

    deals: list[DealRecord] = []
    for i in range(n):
        acquirer = _synth_name(rng, _ACQ_PREFIX, used_names)
        target = _synth_name(rng, _TGT_PREFIX, used_names)
        sector = _SECTORS[i % len(_SECTORS)]
        accession = f"0000{900000 + i}-20-{i:06d}"
        ann = dt.date(2020, 1, 6) + dt.timedelta(days=int(rng.randrange(0, 240)))

        price = round(rng.uniform(8.0, 240.0), 2)
        premium = round(rng.uniform(8.0, 62.0), 1)
        consideration = rng.choice(_CONSIDERATION)
        close = rng.choice(_QUARTERS)
        size = float(round(rng.lognormvariate(13.0, 1.6), 2))

        deal_key = f"{accession}:{target}"

        def mk(
            name: str, raw: str, norm: str, key: str = deal_key, acc: str = accession
        ) -> ProtectedField:
            return ProtectedField(
                field_id=short_hash(key, name),
                name=name,  # type: ignore[arg-type]
                value_raw=raw,
                value_normalized=norm,
                source_accession=acc,
                source_span=(0, 0),
            )

        fields = (
            mk("offer_price_per_share", f"{price:.2f}", f"{price:.10g}"),
            mk("premium_pct", f"{premium:.1f}", f"{premium:.10g}"),
            mk(
                "consideration_type",
                consideration,
                consideration.replace(" ", "_").replace("-", "_"),
            ),
            mk("expected_close", close, close.lower()),
            mk("announcement_date", ann.isoformat(), ann.isoformat()),
        )

        deals.append(
            DealRecord(
                deal_id=short_hash(deal_key),
                acquirer_name=acquirer,
                target_name=target,
                target_cik=f"{1000000 + i}",
                announcement_date=ann,
                sic_2digit=sector,
                size_metric_value=size,
                size_band=0,
                protected_fields=fields,
                source_accessions=(accession,),
            )
        )

    n_controls = max(n, int(round(n * cfg.control_ratio)))
    candidates = [
        Candidate(
            cik=f"{5000000 + j}",
            name=_synth_name(rng, _TGT_PREFIX + _ACQ_PREFIX, used_names),
            sic_2digit=_SECTORS[j % len(_SECTORS)],
            size_metric_value=float(round(rng.lognormvariate(13.0, 1.6), 2)),
        )
        for j in range(n_controls + n)  # slack so matching has room to be picky
    ]

    pairs = match_controls(deals, candidates, sic_digits=cfg.sic_match_digits)
    entities, banded = build_entities(pairs)
    return _assemble(
        banded, entities, [], {"mode": "synthetic", "seed": cfg.synthetic_seed, "n": n}
    )


# Live corpus


def build_live_corpus(cfg: CorpusConfig, cache_dir: Path) -> Corpus:
    """Pull real filings from EDGAR and build the corpus from them."""
    client = EdgarClient(cfg.user_agent, DiskHttpCache(cache_dir / "http"))

    rows: list[IndexRow] = []
    for quarter in cfg.quarters:
        rows.extend(client.form_index(quarter, cfg.form_types))
    rows.sort(key=lambda r: (r.date_filed, r.accession))

    deals: list[DealRecord] = []
    exclusions: list[ExclusionRecord] = []
    seen_keys: set[str] = set()
    attempted = 0
    fetch_failures = 0

    # Cap the candidate scan so a wide quarter range cannot run away. The multiplier
    # is generous because the hit rate on 8-K merger extraction is low.
    budget = max(200, cfg.target_restricted * 60)

    for row in rows[:budget]:
        if len(deals) >= cfg.target_restricted:
            break
        attempted += 1
        try:
            raw = client.fetch(f"{EDGAR_ARCHIVES}/{row.file_name}").decode(
                "latin-1", errors="replace"
            )
        except CorpusFetchError as exc:
            fetch_failures += 1
            exclusions.append(ExclusionRecord(row.accession, "fetch_failed", str(exc)))
            if attempted >= 20 and fetch_failures / attempted > cfg.max_fetch_failure_rate:
                raise CorpusFetchError(
                    f"fetch failure rate {fetch_failures}/{attempted} exceeds "
                    f"{cfg.max_fetch_failure_rate}, aborting rather than shipping a thin corpus"
                ) from exc
            continue

        filed = dt.date.fromisoformat(row.date_filed)
        parsed = parse_or_exclude(
            raw,
            row.accession,
            filed,
            row.cik,
            company_name_hint=row.company_name,
        )
        if parsed.exclusion is not None:
            exclusions.append(parsed.exclusion)
            continue

        rec = parsed.record
        assert rec is not None
        key = f"{rec.acquirer_name}|{rec.target_name}"
        if key in seen_keys:
            exclusions.append(ExclusionRecord(row.accession, "duplicate_deal", key))
            continue
        seen_keys.add(key)

        size = _size_from_facts(client, rec.target_cik, cfg.size_metric)
        deals.append(
            DealRecord(
                deal_id=rec.deal_id,
                acquirer_name=rec.acquirer_name,
                acquirer_cik=rec.acquirer_cik,
                target_name=rec.target_name,
                target_cik=rec.target_cik,
                announcement_date=rec.announcement_date,
                sic_2digit=rec.sic_2digit,
                size_metric_value=size,
                size_band=0,
                protected_fields=rec.protected_fields,
                source_accessions=rec.source_accessions,
            )
        )

    if len(deals) < cfg.target_restricted:
        raise MatchingInfeasibleError(
            f"only extracted {len(deals)} deals from {attempted} candidate filings, "
            f"need {cfg.target_restricted}. Widen corpus.quarters and rerun. "
            f"The HTTP cache is kept, so the rerun will not refetch."
        )

    candidates = [
        Candidate(
            cik=r.cik,
            name=r.company_name,
            sic_2digit="99",
            size_metric_value=0.0,
        )
        for r in rows
        if not any(h in r.form_type.upper() for h in _MA_FORM_HINTS)
    ]
    pairs = match_controls(deals, candidates, sic_digits=cfg.sic_match_digits)
    entities, banded = build_entities(pairs)

    return _assemble(
        banded,
        entities,
        exclusions,
        {
            "mode": "live",
            "quarters": list(cfg.quarters),
            "form_types": list(cfg.form_types),
            "target_restricted": cfg.target_restricted,
        },
    )


EDGAR_ARCHIVES = "https://www.sec.gov/Archives"


def _size_from_facts(client: EdgarClient, cik: str, concept: str) -> float:
    """Most recent annual value of an XBRL concept, or 0 when unavailable."""
    facts = client.company_facts(cik)
    try:
        units = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except (KeyError, TypeError):
        return 0.0
    annual = [u for u in units if u.get("form") in ("10-K", "20-F") and "val" in u]
    if not annual:
        return 0.0
    latest = max(annual, key=lambda u: str(u.get("end", "")))
    try:
        return float(latest["val"])
    except (TypeError, ValueError):
        return 0.0


def build_corpus(cfg: SilentwallConfig) -> Corpus:
    """Entry point. Routes to synthetic or live based on config."""
    if cfg.corpus.synthetic:
        return build_synthetic_corpus(cfg.corpus)
    return build_live_corpus(cfg.corpus, cfg.data_dir)


def build_artifacts(corpus: Corpus, seed: int) -> list[PrivateArtifact]:
    """Private-side artifacts for every deal in the corpus."""
    out: list[PrivateArtifact] = []
    for deal in corpus.deals:
        out.extend(synthesize_artifacts(deal, seed))
    out.sort(key=lambda a: a.artifact_id)
    return out
