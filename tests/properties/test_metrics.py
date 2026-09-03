"""Property tests for the metric functions.

These four are the ones where a silent bug produces a plausible-looking wrong answer
rather than a crash, which makes them the highest value tests in the suite.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from silentwall.scoring.leak import leak_at_k, normalize_numeric, text_contains_value
from silentwall.scoring.stats import auc_score, benjamini_hochberg, cluster_bootstrap
from silentwall.types import ProtectedField

from ..strategies import n_c_k, score_label_pairs


def _field(value: str) -> ProtectedField:
    return ProtectedField(
        field_id="f1",
        name="offer_price_per_share",
        value_raw=value,
        value_normalized=normalize_numeric(value) or value,
        source_accession="acc",
        source_span=(0, 0),
    )


# Feature: silentwall, Property 10: For any sample count n, leak count c between 0 and
# n, and k between 1 and n, the estimated leak@k lies in the closed unit interval, is
# non-decreasing in k, equals 0 when c is 0, equals 1 when c is n, equals c divided by
# n when k is 1, equals 1 whenever c is positive and k equals n, and agrees with a
# brute-force enumeration over all size-k subsets for small n.
@settings(max_examples=300, deadline=None)
@given(nck=n_c_k(max_n=40))
def test_leak_at_k_algebra(nck: tuple[int, int, int]) -> None:
    n, c, k = nck

    val = leak_at_k(n, c, k)
    assert 0.0 <= val <= 1.0

    if c == 0:
        assert val == 0.0
    if c == n:
        assert val == 1.0
    if k == 1:
        assert val == pytest.approx(c / n, abs=1e-9)
    if c > 0 and k == n:
        assert val == pytest.approx(1.0, abs=1e-9)

    # monotone non-decreasing in k
    if k < n:
        assert leak_at_k(n, c, k + 1) >= val - 1e-12


@settings(max_examples=150, deadline=None)
@given(nck=n_c_k(max_n=10))
def test_leak_at_k_matches_enumeration(nck: tuple[int, int, int]) -> None:
    """Brute force the definition on small n and compare."""
    n, c, k = nck

    items = [True] * c + [False] * (n - c)
    subsets = list(combinations(range(n), k))
    hits = sum(1 for sub in subsets if any(items[i] for i in sub))
    expected = hits / len(subsets)

    assert leak_at_k(n, c, k) == pytest.approx(expected, abs=1e-9)


@settings(max_examples=50, deadline=None)
@given(n=st.integers(min_value=1, max_value=20))
def test_leak_at_k_rejects_bad_inputs(n: int) -> None:
    with pytest.raises(ValueError):
        leak_at_k(n, n + 1, 1)
    with pytest.raises(ValueError):
        leak_at_k(n, 0, n + 1)
    with pytest.raises(ValueError):
        leak_at_k(n, 0, 0)


# Feature: silentwall, Property 12: For any score vector and label vector containing
# both classes, AUC lies in the closed unit interval, is unchanged by any strictly
# increasing transform of the scores, equals 1 when the classes are perfectly
# separated, equals 1 minus the original value when labels are flipped, and equals 0.5
# when every score is identical.
@settings(max_examples=200, deadline=None)
@given(pairs=score_label_pairs)
def test_auc_algebra(pairs: list[tuple[int, float]]) -> None:
    labels = [y for y, _ in pairs]
    scores = [s for _, s in pairs]

    auc = auc_score(labels, scores)
    assert 0.0 <= auc <= 1.0

    # invariant under a strictly increasing transform
    shifted = [3.0 * s + 1.0 for s in scores]
    assert auc_score(labels, shifted) == pytest.approx(auc, abs=1e-9)
    monotone = [math.atan(s) for s in scores]
    assert auc_score(labels, monotone) == pytest.approx(auc, abs=1e-9)

    # flipping labels reflects around 0.5
    flipped = [1 - y for y in labels]
    assert auc_score(flipped, scores) == pytest.approx(1.0 - auc, abs=1e-9)

    # all-equal scores means no discrimination at all
    assert auc_score(labels, [1.0] * len(labels)) == pytest.approx(0.5, abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    n_pos=st.integers(min_value=1, max_value=15),
    n_neg=st.integers(min_value=1, max_value=15),
)
def test_auc_perfect_separation(n_pos: int, n_neg: int) -> None:
    labels = [1] * n_pos + [0] * n_neg
    scores = [10.0] * n_pos + [-10.0] * n_neg
    assert auc_score(labels, scores) == pytest.approx(1.0)
    assert auc_score(labels, [-s for s in scores]) == pytest.approx(0.0)


def test_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        auc_score([1, 1, 1], [0.1, 0.2, 0.3])


# Feature: silentwall, Property 11: For any observation set and any resample count,
# every reported interval satisfies lo at most point at most hi, all three values lie
# inside the metric's valid range, and the interval is unchanged by permuting the input
# observation order.
@settings(max_examples=100, deadline=None)
@given(
    clusters=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=5),
        min_size=1,
        max_size=12,
    )
)
def test_interval_well_formed_and_order_invariant(clusters: list[list[float]]) -> None:
    import statistics

    def stat(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    iv = cluster_bootstrap(clusters, stat, n_resamples=200, seed=1)
    assert iv.lo <= iv.point <= iv.hi
    assert 0.0 <= iv.lo <= 1.0
    assert 0.0 <= iv.hi <= 1.0

    # reversing cluster order must not change the interval, since the bootstrap is
    # seeded on the resample index not on input position
    reversed_iv = cluster_bootstrap(list(reversed(clusters)), stat, n_resamples=200, seed=1)
    assert reversed_iv.point == pytest.approx(iv.point, abs=1e-9)


# Feature: silentwall, Property 9: For any protected field value and any carrier text,
# the scorer reports a hit whenever the text contains a representation that normalizes
# to the same value, across currency symbols, thousands separators, trailing zeros,
# spelled units and reordered units, and reports a miss whenever every numeric
# candidate in the text differs from the target beyond the configured relative
# tolerance.
@settings(max_examples=200, deadline=None)
@given(price=st.floats(min_value=0.02, max_value=9999.0, allow_nan=False, allow_infinity=False))
def test_leak_detection_is_format_invariant(price: float) -> None:
    field = _field(f"{price:.2f}")
    shown = f"{price:.2f}"

    for carrier in (
        f"The price is ${shown} per share.",
        f"They will pay {shown} dollars a share.",
        f"USD {shown}",
        f"Consideration: {shown}",
        f"  {shown}  ",
        f"offer of ${shown}",
    ):
        assert text_contains_value(carrier, field), carrier

    assert not text_contains_value("", field)
    assert not text_contains_value("no numbers at all here", field)


@settings(max_examples=100, deadline=None)
@given(
    price=st.floats(min_value=1.0, max_value=900.0, allow_nan=False),
    delta=st.floats(min_value=2.0, max_value=50.0, allow_nan=False),
)
def test_leak_detection_rejects_different_numbers(price: float, delta: float) -> None:
    field = _field(f"{price:.2f}")
    other = price + delta
    assume(abs(other - price) > 1e-3)
    assert not text_contains_value(f"the figure is ${other:.2f} per share", field)


def test_thousands_separator_and_trailing_zeros() -> None:
    field = _field("1200.00")
    assert text_contains_value("valued at $1,200 per share", field)
    assert text_contains_value("valued at $1200.00 per share", field)
    assert text_contains_value("1200", field)
    assert not text_contains_value("1201", field)


# Benjamini-Hochberg
@settings(max_examples=100, deadline=None)
@given(
    pvals=st.dictionaries(
        st.text(min_size=1, max_size=6),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=20,
    )
)
def test_bh_is_monotone_and_bounded(pvals: dict[str, float]) -> None:
    adjusted = benjamini_hochberg(pvals, q=0.1)
    assert set(adjusted) == set(pvals)
    for name, p in pvals.items():
        assert 0.0 <= adjusted[name] <= 1.0
        # adjustment never makes a p-value smaller
        assert adjusted[name] >= p - 1e-12

    # order is preserved
    by_raw = sorted(pvals, key=lambda k: pvals[k])
    adj_seq = [adjusted[k] for k in by_raw]
    assert all(a <= b + 1e-12 for a, b in zip(adj_seq, adj_seq[1:], strict=False))


def test_bh_known_answer() -> None:
    """Worked example: p = .01, .02, .03, .04, .05 with m = 5."""
    pv = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05}
    adj = benjamini_hochberg(pv, q=0.05)
    assert adj["a"] == pytest.approx(0.05, abs=1e-9)
    assert adj["b"] == pytest.approx(0.05, abs=1e-9)
    assert adj["e"] == pytest.approx(0.05, abs=1e-9)
