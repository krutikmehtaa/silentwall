"""Build the control group.

Controls are the negative class for detectability. Without them the AUC question is
meaningless: "can you tell restricted entities apart" needs something to tell them
apart from, and the something has to resemble them, otherwise the classifier learns
sector or size instead of the behavioural signature we are actually measuring.

Matching is on sector and size band, greedy nearest neighbour without replacement,
iterating restricted entities in sorted order so the result does not depend on
dictionary iteration order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import MatchingInfeasibleError
from ..hashing import short_hash
from ..types import DealRecord, Entity

__all__ = ["Candidate", "assign_size_bands", "match_controls", "build_entities"]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A company eligible to serve as a control."""

    cik: str
    name: str
    sic_2digit: str
    size_metric_value: float


def assign_size_bands(values: Sequence[float], n_bands: int = 5) -> list[int]:
    """Bucket values into quantile bands of their logarithm.

    Log first because company size is heavy tailed: a linear bucketing would put
    almost everything in the bottom band. Ties go to the lower band so the mapping
    stays a pure function of the input.
    """
    if not values:
        return []
    if n_bands < 1:
        raise ValueError("n_bands must be at least 1")

    logs = [math.log1p(max(0.0, v)) for v in values]
    order = sorted(range(len(logs)), key=lambda i: (logs[i], i))
    bands = [0] * len(logs)
    for rank, idx in enumerate(order):
        bands[idx] = min(n_bands - 1, (rank * n_bands) // len(order))
    return bands


def match_controls(
    deals: Sequence[DealRecord],
    candidates: Sequence[Candidate],
    n_bands: int = 5,
    sic_digits: int = 2,
) -> list[tuple[DealRecord, Candidate]]:
    """Pair each deal with one control on (sector, size band), no reuse.

    Falls back to sector-only, then to nearest size anywhere, before giving up.
    Raising when a deal cannot be matched is deliberate: shipping fewer pairs than
    configured would silently widen every interval in the report.
    """
    if not deals:
        return []

    excluded_ciks = {d.target_cik for d in deals} | {
        d.acquirer_cik for d in deals if d.acquirer_cik
    }
    pool = [c for c in candidates if c.cik not in excluded_ciks]
    if len(pool) < len(deals):
        raise MatchingInfeasibleError(
            f"need {len(deals)} controls, only {len(pool)} eligible candidates available"
        )

    all_values = [d.size_metric_value for d in deals] + [c.size_metric_value for c in pool]
    all_bands = assign_size_bands(all_values, n_bands)
    deal_bands = {d.deal_id: all_bands[i] for i, d in enumerate(deals)}
    cand_bands = {c.cik: all_bands[len(deals) + i] for i, c in enumerate(pool)}

    used: set[str] = set()
    pairs: list[tuple[DealRecord, Candidate]] = []

    for deal in sorted(deals, key=lambda d: d.deal_id):
        want_sector = deal.sic_2digit[:sic_digits]
        want_band = deal_bands[deal.deal_id]

        tiers = (
            [
                c
                for c in pool
                if c.sic_2digit[:sic_digits] == want_sector and cand_bands[c.cik] == want_band
            ],
            [c for c in pool if c.sic_2digit[:sic_digits] == want_sector],
            list(pool),
        )

        chosen: Candidate | None = None
        for tier in tiers:
            avail = [c for c in tier if c.cik not in used]
            if avail:
                target_log = math.log1p(max(0.0, deal.size_metric_value))
                chosen = min(
                    avail,
                    key=lambda c: (
                        abs(math.log1p(max(0.0, c.size_metric_value)) - target_log),
                        c.cik,
                    ),
                )
                break

        if chosen is None:
            raise MatchingInfeasibleError(f"no control available for deal {deal.deal_id}")

        used.add(chosen.cik)
        pairs.append((deal, chosen))

    return pairs


def build_entities(
    pairs: Sequence[tuple[DealRecord, Candidate]],
    n_bands: int = 5,
) -> tuple[list[Entity], list[DealRecord]]:
    """Turn matched pairs into Entity records with a shared pair_id.

    Also returns the deals with their size_band filled in, since banding needs the
    whole cohort and cannot be done at parse time.
    """
    if not pairs:
        return [], []

    values = [d.size_metric_value for d, _ in pairs] + [c.size_metric_value for _, c in pairs]
    bands = assign_size_bands(values, n_bands)
    n = len(pairs)

    entities: list[Entity] = []
    banded_deals: list[DealRecord] = []

    for i, (deal, cand) in enumerate(pairs):
        pair_id = short_hash("pair", deal.deal_id)
        deal_band = bands[i]
        ctrl_band = bands[n + i]

        banded = DealRecord(
            deal_id=deal.deal_id,
            acquirer_name=deal.acquirer_name,
            acquirer_cik=deal.acquirer_cik,
            target_name=deal.target_name,
            target_cik=deal.target_cik,
            announcement_date=deal.announcement_date,
            sic_2digit=deal.sic_2digit,
            size_metric_value=deal.size_metric_value,
            size_band=deal_band,
            protected_fields=deal.protected_fields,
            source_accessions=deal.source_accessions,
        )
        banded_deals.append(banded)

        entities.append(
            Entity(
                entity_id=short_hash("ent", deal.deal_id, "restricted"),
                display_name=deal.target_name,
                cik=deal.target_cik,
                entity_class="restricted",
                sic_2digit=deal.sic_2digit,
                size_band=deal_band,
                pair_id=pair_id,
                deal_id=deal.deal_id,
            )
        )
        entities.append(
            Entity(
                entity_id=short_hash("ent", cand.cik, "control", pair_id),
                display_name=cand.name,
                cik=cand.cik,
                entity_class="control",
                sic_2digit=cand.sic_2digit,
                size_band=ctrl_band,
                pair_id=pair_id,
                deal_id=None,
            )
        )

    entities.sort(key=lambda e: e.entity_id)
    banded_deals.sort(key=lambda d: d.deal_id)
    return entities, banded_deals
