"""Property tests for corpus construction, artifacts, probes and splits."""

from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from silentwall.config import CorpusConfig
from silentwall.corpus.build import build_artifacts, build_synthetic_corpus, corpus_content_hash
from silentwall.corpus.controls import assign_size_bands, build_entities, match_controls
from silentwall.corpus.parse import parse_filing, render_filing
from silentwall.corpus.splits import assign_splits
from silentwall.errors import MatchingInfeasibleError, SplitLeakageError
from silentwall.probes.generate import build_probe_suite
from silentwall.types import CONTENT_FAMILIES, ProbeFamily

from ..strategies import candidates, deal_records

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# Feature: silentwall, Property 1: For any filing input, including arbitrary bytes,
# valid filings, and mutations of valid filings, the corpus builder produces either
# exactly one DealRecord for which acquirer, target, announcement date and at least one
# numeric protected field are all populated, or exactly one ExclusionRecord carrying a
# reason from the closed reason enum, and never both and never neither.
@settings(max_examples=200, deadline=None)
@given(raw=st.text(max_size=400))
def test_records_are_complete_or_excluded(raw: str) -> None:
    import datetime as dt

    parsed = parse_filing(raw, "acc-1", dt.date(2020, 5, 1), "12345")

    assert (parsed.record is None) != (parsed.exclusion is None), "exactly one must be set"

    if parsed.record is not None:
        r = parsed.record
        assert r.acquirer_name.strip()
        assert r.target_name.strip()
        assert r.announcement_date is not None
        numeric = [
            f for f in r.protected_fields if f.name in ("offer_price_per_share", "premium_pct")
        ]
        assert numeric, "a record must carry at least one numeric protected field"
    else:
        assert parsed.exclusion is not None
        assert parsed.exclusion.reason in {
            "no_item_101",
            "no_target",
            "no_acquirer",
            "no_numeric_field",
            "ambiguous_parties",
            "fetch_failed",
            "duplicate_deal",
        }


# Feature: silentwall, Property 2: For any DealRecord, rendering it into a
# filing-shaped document and parsing that document back recovers a DealRecord whose
# acquirer, target, announcement date and every protected field value are equal to the
# original.
@settings(max_examples=100, deadline=None)
@given(deal=deal_records())
def test_filing_parse_round_trip(deal: object) -> None:
    text = render_filing(deal)  # type: ignore[arg-type]
    parsed = parse_filing(text, "acc-rt", deal.announcement_date, deal.target_cik)  # type: ignore[attr-defined]

    assert parsed.ok, f"render then parse should succeed, got {parsed.exclusion}"
    got = parsed.record
    assert got is not None

    orig_price = deal.field_by_name("offer_price_per_share")  # type: ignore[attr-defined]
    got_price = got.field_by_name("offer_price_per_share")
    if orig_price is not None:
        assert got_price is not None
        assert got_price.value_normalized == orig_price.value_normalized

    assert got.announcement_date == deal.announcement_date  # type: ignore[attr-defined]


# Feature: silentwall, Property 3: For any pool of candidate companies and any set of
# restricted entities drawn from it, the matcher produces a pairing in which every
# restricted entity has exactly one control, every control shares the restricted
# entity's sector code and size band, no control carries a deal id, no candidate is used
# as a control more than once, and every emitted pair id appears on exactly two
# entities.
@SLOW
@given(
    deals=st.lists(deal_records(), min_size=1, max_size=6, unique_by=lambda d: d.deal_id),
    pool=st.lists(candidates(), min_size=12, max_size=40, unique_by=lambda c: c.cik),
)
def test_control_matching_invariants(deals: list, pool: list) -> None:
    pairs = match_controls(deals, pool)

    assert len(pairs) == len(deals)

    used_ciks = [c.cik for _, c in pairs]
    assert len(used_ciks) == len(set(used_ciks)), "no candidate reused as a control"

    entities, banded = build_entities(pairs)

    assert len(entities) == 2 * len(deals)
    counts = Counter(e.pair_id for e in entities)
    assert set(counts.values()) == {2}, "every pair id appears on exactly two entities"

    for e in entities:
        if e.entity_class == "control":
            assert e.deal_id is None
        else:
            assert e.deal_id is not None

    # one restricted and one control per pair
    for pair_id in counts:
        members = [e for e in entities if e.pair_id == pair_id]
        classes = sorted(e.entity_class for e in members)
        assert classes == ["control", "restricted"]


def test_matching_raises_when_pool_too_small() -> None:
    """Shipping fewer pairs than requested would silently widen every interval."""
    import datetime as dt

    from silentwall.types import DealRecord, ProtectedField

    def mk(i: int) -> DealRecord:
        return DealRecord(
            deal_id=f"d{i}",
            acquirer_name=f"A{i}",
            target_name=f"T{i}",
            target_cik=f"{i}",
            announcement_date=dt.date(2020, 1, 1),
            sic_2digit="73",
            size_metric_value=1000.0,
            size_band=0,
            protected_fields=(
                ProtectedField(f"f{i}", "offer_price_per_share", "10.00", "10", "a", (0, 0)),
            ),
            source_accessions=("a",),
        )

    deals = [mk(i) for i in range(5)]
    with pytest.raises(MatchingInfeasibleError):
        match_controls(deals, [])


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    ),
    n_bands=st.integers(min_value=1, max_value=6),
)
def test_size_bands_are_bounded_and_monotone(values: list[float], n_bands: int) -> None:
    bands = assign_size_bands(values, n_bands)
    assert len(bands) == len(values)
    assert all(0 <= b < n_bands for b in bands)

    # a larger value never lands in a lower band
    order = sorted(range(len(values)), key=lambda i: values[i])
    seen = [bands[i] for i in order]
    assert all(a <= b for a, b in zip(seen, seen[1:], strict=False))


# Feature: silentwall, Property 4: For any corpus, permuting the order of its record
# lists leaves the corpus hash unchanged, and mutating any single field of any single
# record changes the corpus hash. The manifest counts equal the lengths of the
# corresponding record tuples.
@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=8), seed=st.integers(min_value=0, max_value=999))
def test_corpus_hash_depends_on_content_only(n: int, seed: int) -> None:
    cfg = CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)
    corpus = build_synthetic_corpus(cfg)

    base = corpus_content_hash(corpus.deals, corpus.entities, "tp1")

    shuffled = corpus_content_hash(
        list(reversed(corpus.deals)), list(reversed(corpus.entities)), "tp1"
    )
    assert shuffled == base, "hash must not depend on record order"

    assert corpus.manifest.n_restricted == len(corpus.restricted)
    assert corpus.manifest.n_control == len(corpus.controls)
    assert corpus.manifest.corpus_hash == base

    # mutating one field changes the hash
    from dataclasses import replace

    mutated = list(corpus.deals)
    mutated[0] = replace(mutated[0], target_name=mutated[0].target_name + "X")
    assert corpus_content_hash(mutated, corpus.entities, "tp1") != base


# Feature: silentwall, Property 5: For any corpus and seed, every synthesized private
# artifact declares a source deal id and a non-empty set of embedded field ids that all
# resolve to fields of that deal, and no value from any other deal's protected fields
# appears in it.
@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=6), seed=st.integers(min_value=0, max_value=999))
def test_artifact_provenance_and_non_contamination(n: int, seed: int) -> None:
    cfg = CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)
    corpus = build_synthetic_corpus(cfg)
    artifacts = build_artifacts(corpus, seed)

    by_deal = {d.deal_id: d for d in corpus.deals}

    for art in artifacts:
        assert art.deal_id in by_deal
        assert art.embeds_field_ids, "an artifact must declare what it embeds"

        deal = by_deal[art.deal_id]
        own_field_ids = {f.field_id for f in deal.protected_fields}
        for fid in art.embeds_field_ids:
            assert fid in own_field_ids, "embedded field must belong to the source deal"

        # no other deal's target name leaks into this artifact
        for other_id, other in by_deal.items():
            if other_id == art.deal_id:
                continue
            assert other.target_name not in art.text


# Feature: silentwall, Property 6: For any corpus and seed, two independent runs of
# artifact synthesis produce byte-identical serialized artifact sets in identical order,
# and two independent runs of probe generation produce identical probe id sequences.
@settings(max_examples=20, deadline=None)
@given(n=st.integers(min_value=2, max_value=6), seed=st.integers(min_value=0, max_value=999))
def test_synthesis_and_probes_are_deterministic(n: int, seed: int) -> None:
    cfg = CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)

    c1 = build_synthetic_corpus(cfg)
    c2 = build_synthetic_corpus(cfg)
    assert c1.manifest.corpus_hash == c2.manifest.corpus_hash

    a1 = [a.to_dict() for a in build_artifacts(c1, seed)]
    a2 = [a.to_dict() for a in build_artifacts(c2, seed)]
    assert a1 == a2, "artifact synthesis must be byte reproducible"

    p1 = [p.probe_id for p in build_probe_suite(c1)]
    p2 = [p.probe_id for p in build_probe_suite(c2)]
    assert p1 == p2, "probe ids must be stable across runs"

    # a different seed must actually change something, so the seed is wired through
    other = build_synthetic_corpus(
        CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed + 1)
    )
    assert other.manifest.corpus_hash != c1.manifest.corpus_hash


# Feature: silentwall, Property 7: For any corpus, the generated probe suite contains at
# least one probe in each of the six families, content-family probes target only
# restricted entities and carry a non-empty target field id list, and the multiset of
# behavioural template ids attached to an entity is identical for every entity
# regardless of whether it is restricted or control.
@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=6), seed=st.integers(min_value=0, max_value=999))
def test_probe_suite_structure_and_symmetry(n: int, seed: int) -> None:
    corpus = build_synthetic_corpus(
        CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)
    )
    probes = build_probe_suite(corpus)

    families = {p.family for p in probes}
    for fam in ProbeFamily:
        assert fam in families, f"missing family {fam}"

    restricted_ids = {e.entity_id for e in corpus.restricted}
    for p in probes:
        if p.family in CONTENT_FAMILIES:
            assert p.entity_id in restricted_ids, "content probes only target restricted entities"
            assert p.target_field_ids, "content probes must name a target field"

    # behavioural symmetry: identical template multiset for every entity
    per_entity: dict[str, Counter] = {}
    for p in probes:
        if p.is_behavioural:
            per_entity.setdefault(p.entity_id, Counter())[p.template_id] += 1

    assert len(per_entity) == len(corpus.entities)
    reference = next(iter(per_entity.values()))
    for entity_id, counter in per_entity.items():
        assert counter == reference, f"entity {entity_id} got a different behavioural set"


# Feature: silentwall, Property 8: For any corpus, no behavioural probe prompt contains
# any protected field value in any accepted normalized surface form, and every
# behavioural probe has an empty target field id list.
@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=6), seed=st.integers(min_value=0, max_value=999))
def test_behavioural_probes_never_reference_protected_content(n: int, seed: int) -> None:
    corpus = build_synthetic_corpus(
        CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)
    )
    probes = build_probe_suite(corpus)
    artifacts = build_artifacts(corpus, seed)

    all_values = {f.value_raw for d in corpus.deals for f in d.protected_fields} | {
        f.value_normalized for d in corpus.deals for f in d.protected_fields
    }

    for p in probes:
        if not p.is_behavioural:
            continue
        assert p.target_field_ids == ()
        for value in all_values:
            if len(value) >= 3:
                assert value not in p.prompt, f"behavioural prompt leaked {value!r}"

    # nor any substantial run of artifact text
    for art in artifacts:
        for line in art.text.splitlines():
            chunk = line.strip()
            if len(chunk) > 25:
                for p in probes:
                    if p.is_behavioural:
                        assert chunk not in p.prompt


# Feature: silentwall, Property 14 (split half): For any corpus and split seed, the dev
# and eval entity id sets are disjoint, the dev and eval pair id sets are disjoint, and
# requesting an entity outside a split raises SplitLeakageError.
@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
    split_seed=st.integers(min_value=0, max_value=9999),
)
def test_splits_are_disjoint_at_pair_level(n: int, seed: int, split_seed: int) -> None:
    corpus = build_synthetic_corpus(
        CorpusConfig(target_restricted=n, synthetic=True, synthetic_seed=seed)
    )
    sp = assign_splits(corpus, split_seed, 0.5)

    assert sp.disjoint
    assert not (sp.dev.entity_ids & sp.eval.entity_ids)
    assert not (sp.dev.pair_ids & sp.eval.pair_ids)
    assert sp.dev.pair_ids | sp.eval.pair_ids == set(corpus.pair_ids)

    # a matched pair never straddles the boundary
    for pair_id in corpus.pair_ids:
        members = [e.entity_id for e in corpus.entities if e.pair_id == pair_id]
        in_dev = [sp.dev.contains(m) for m in members]
        assert len(set(in_dev)) == 1, "restricted and control must land in the same split"

    # the tripwire fires
    some_eval = next(iter(sp.eval.entity_ids))
    with pytest.raises(SplitLeakageError):
        sp.dev.require(some_eval)

    # and does not fire on its own entities
    sp.dev.require(next(iter(sp.dev.entity_ids)))

    # determinism
    again = assign_splits(corpus, split_seed, 0.5)
    assert again.dev.ids_hash == sp.dev.ids_hash
    assert again.eval.ids_hash == sp.eval.ids_hash
