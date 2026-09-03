"""Hypothesis strategies.

The generators are where the edge cases live: empty generations, k of one,
all-identical samples, missing logprobs, unicode names, values at the numeric
tolerance boundary, and AUC inputs densely sampled around the 0.6 decision boundary.
"""

from __future__ import annotations

import datetime as dt

from hypothesis import strategies as st

from silentwall.corpus.controls import Candidate
from silentwall.types import (
    DealRecord,
    Entity,
    Generation,
    Probe,
    ProbeFamily,
    ProtectedField,
    TokenTrace,
)

SECTORS = ["20", "28", "35", "36", "48", "60", "73", "87"]

company_names = st.one_of(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll"), min_codepoint=65, max_codepoint=122
        ),
        min_size=3,
        max_size=18,
    ),
    st.sampled_from(["Acme Corp", "Grupo Sol", "Nordbank AS", "Zephyr Labs", "Ostrom AB"]),
)

# Rounded to cents at the source. ProtectedField carries an invariant that
# value_normalized is the normalization of value_raw, so a price with more precision
# than the raw rendering would break that invariant in the fixture rather than in the
# code under test.
prices = st.one_of(
    st.floats(min_value=0.01, max_value=9999.0, allow_nan=False, allow_infinity=False).map(
        lambda x: round(x, 2)
    ),
    st.sampled_from([1.0, 10.0, 34.0, 34.5, 100.0, 0.5, 1200.0]),
).filter(lambda x: x >= 0.01)


@st.composite
def protected_fields(draw: st.DrawFn, deal_key: str = "dk") -> ProtectedField:
    price = draw(prices)
    raw = f"{price:.2f}"
    return ProtectedField(
        field_id=draw(st.text(min_size=3, max_size=10, alphabet="abcdef0123456789")),
        name="offer_price_per_share",
        value_raw=raw,
        value_normalized=f"{price:.10g}",
        source_accession="0000-00-000000",
        source_span=(0, len(raw)),
    )


@st.composite
def deal_records(draw: st.DrawFn) -> DealRecord:
    deal_id = draw(st.text(min_size=4, max_size=12, alphabet="abcdef0123456789"))

    # Acquirer and target must differ. A company does not acquire itself, and the
    # parser correctly rejects that case as ambiguous parties, so generating it would
    # test the strategy rather than the code.
    acquirer = draw(company_names)
    target = draw(company_names.filter(lambda t: t.strip().lower() != acquirer.strip().lower()))

    price = draw(prices)
    premium = draw(st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
    day = draw(st.integers(min_value=0, max_value=900))

    def mk(name: str, raw: str, norm: str) -> ProtectedField:
        return ProtectedField(
            field_id=f"{deal_id}:{name}",
            name=name,  # type: ignore[arg-type]
            value_raw=raw,
            value_normalized=norm,
            source_accession="acc",
            source_span=(0, 0),
        )

    ann = dt.date(2019, 1, 1) + dt.timedelta(days=day)
    return DealRecord(
        deal_id=deal_id,
        acquirer_name=acquirer,
        target_name=target,
        target_cik=draw(st.text(min_size=4, max_size=8, alphabet="0123456789")),
        announcement_date=ann,
        sic_2digit=draw(st.sampled_from(SECTORS)),
        size_metric_value=draw(
            st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)
        ),
        size_band=0,
        protected_fields=(
            mk("offer_price_per_share", f"{price:.2f}", f"{price:.10g}"),
            mk("premium_pct", f"{premium:.1f}", f"{premium:.10g}"),
            mk("announcement_date", ann.isoformat(), ann.isoformat()),
        ),
        source_accessions=("acc",),
    )


@st.composite
def candidates(draw: st.DrawFn) -> Candidate:
    return Candidate(
        cik=draw(st.text(min_size=5, max_size=9, alphabet="0123456789")),
        name=draw(company_names),
        sic_2digit=draw(st.sampled_from(SECTORS)),
        size_metric_value=draw(
            st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def entities(draw: st.DrawFn, entity_class: str | None = None) -> Entity:
    cls = entity_class or draw(st.sampled_from(["restricted", "control"]))
    pair_id = draw(st.text(min_size=4, max_size=10, alphabet="abcdef0123456789"))
    return Entity(
        entity_id=draw(st.text(min_size=4, max_size=12, alphabet="abcdef0123456789")),
        display_name=draw(company_names),
        cik=draw(st.text(min_size=4, max_size=8, alphabet="0123456789")),
        entity_class=cls,  # type: ignore[arg-type]
        sic_2digit=draw(st.sampled_from(SECTORS)),
        size_band=draw(st.integers(min_value=0, max_value=4)),
        pair_id=pair_id,
        deal_id=draw(st.text(min_size=4, max_size=10, alphabet="abcdef0123456789"))
        if cls == "restricted"
        else None,
    )


generation_texts = st.one_of(
    st.just(""),
    st.just("   "),
    st.just("I cannot discuss that."),
    st.just("Based on public filings, margins are near the sector median."),
    st.text(min_size=0, max_size=160),
)


@st.composite
def token_traces(draw: st.DrawFn, n: int) -> TokenTrace | None:
    if draw(st.booleans()):
        return None
    lp = st.floats(min_value=-12.0, max_value=-0.0001, allow_nan=False, allow_infinity=False)
    return TokenTrace(
        token_ids=tuple(draw(st.lists(st.integers(0, 49999), min_size=n, max_size=n))),
        chosen_logprobs=tuple(draw(st.lists(lp, min_size=n, max_size=n))),
        topm_logprobs=tuple(tuple(draw(st.lists(lp, min_size=3, max_size=3))) for _ in range(n)),
    )


@st.composite
def generations(draw: st.DrawFn, probe_id: str = "p", entity_id: str = "e") -> Generation:
    text = draw(generation_texts)
    n = max(1, len(text.split()))
    return Generation(
        cache_key=draw(st.text(min_size=8, max_size=16, alphabet="abcdef0123456789")),
        probe_id=probe_id,
        entity_id=entity_id,
        sample_index=draw(st.integers(min_value=0, max_value=15)),
        seed=draw(st.integers(min_value=0, max_value=2**31)),
        text=text,
        n_tokens=n,
        backend_fp="bfp",
        containment_fp="cfp",
        trace=draw(token_traces(min(n, 6))),
        tool_calls=tuple(draw(st.lists(st.sampled_from(["search_docs", "calc"]), max_size=4))),
        latency_ms=draw(st.floats(min_value=0.0, max_value=5000.0, allow_nan=False)),
        latency_trustworthy=draw(st.booleans()),
    )


@st.composite
def generation_groups(draw: st.DrawFn, min_k: int = 1, max_k: int = 8) -> list[Generation]:
    """k generations for one probe, with distinct sample indices."""
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    identical = draw(st.booleans())
    if identical:
        text = draw(generation_texts)
        texts = [text] * k
    else:
        texts = [draw(generation_texts) for _ in range(k)]

    out: list[Generation] = []
    for i, text in enumerate(texts):
        n = max(1, len(text.split()))
        out.append(
            Generation(
                cache_key=f"k{i}",
                probe_id="p",
                entity_id="e",
                sample_index=i,
                seed=1000 + i,
                text=text,
                n_tokens=n,
                backend_fp="bfp",
                containment_fp="cfp",
                trace=draw(token_traces(min(n, 5))),
                tool_calls=(),
                latency_ms=float(10 + i),
                latency_trustworthy=True,
            )
        )
    return out


@st.composite
def probes(draw: st.DrawFn, family: ProbeFamily | None = None) -> Probe:
    fam = family or draw(st.sampled_from(list(ProbeFamily)))
    behavioural = fam is ProbeFamily.BEHAVIOURAL
    return Probe(
        probe_id=draw(st.text(min_size=4, max_size=12, alphabet="abcdef0123456789")),
        family=fam,
        entity_id=draw(st.text(min_size=4, max_size=10, alphabet="abcdef0123456789")),
        template_id=draw(st.sampled_from(["b1", "b2", "d1", "p1", "i1"])),
        prompt=draw(st.text(min_size=1, max_size=80)),
        target_field_ids=() if behavioural else ("f1",),
        tools_exposed=() if behavioural else tuple(draw(st.lists(st.just("calc"), max_size=1))),
        memory_primed=False if behavioural else draw(st.booleans()),
    )


# Scores are rounded to six decimals on purpose. Unrounded floats let Hypothesis
# produce denormals like 5e-324, and then a mathematically strictly-increasing map such
# as 3x+1 collapses distinct scores to the same float. That breaks rank invariance in
# the test without saying anything about the AUC implementation, which is the quantity
# under test.
_scores = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False).map(
    lambda x: round(x, 6)
)

score_label_pairs = st.lists(
    st.tuples(st.integers(min_value=0, max_value=1), _scores),
    min_size=4,
    max_size=60,
).filter(lambda xs: len({y for y, _ in xs}) == 2)


@st.composite
def n_c_k(draw: st.DrawFn, max_n: int = 40) -> tuple[int, int, int]:
    """Draw (n, c, k) with c and k already inside their valid ranges.

    Generating them independently and filtering rejects most candidates, which trips
    Hypothesis's filter health check and starves the test of examples.
    """
    n = draw(st.integers(min_value=1, max_value=max_n))
    c = draw(st.integers(min_value=0, max_value=n))
    k = draw(st.integers(min_value=1, max_value=n))
    return n, c, k
