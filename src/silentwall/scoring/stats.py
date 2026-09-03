"""Statistics.

Small samples plus many metrics is the standard way research code produces confident
nonsense, so the analysis is fixed here rather than chosen after seeing results.

The unit of analysis is the matched pair, not the probe and not the generation. Probes
within an entity share the entity, the artifacts and the model state, so treating 840
probes as 840 independent observations would shrink every interval by roughly the
square root of the cluster size. Every interval in this file therefore resamples
pairs, not rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ..types import Interval

__all__ = [
    "auc_score",
    "cluster_bootstrap",
    "bootstrap_mean",
    "benjamini_hochberg",
    "permutation_test_auc",
    "power_note",
    "paired_cluster_bootstrap",
]


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, computed by rank with explicit tie handling.

    Written out rather than delegated so ties get midranks, which matters here: a
    defense that collapses every answer to the same output produces all-equal scores,
    and the correct answer in that case is exactly 0.5.
    """
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = midrank
        i = j + 1

    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cluster_bootstrap(
    clusters: Sequence[Sequence[float]],
    statistic: Callable[[Sequence[float]], float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap that resamples clusters with replacement.

    clusters is a list of per-cluster observation lists. Resampling whole clusters is
    what makes the interval honest under within-cluster correlation.
    """
    flat = [v for c in clusters for v in c]
    if not flat:
        return Interval(point=0.0, lo=0.0, hi=0.0, level=level, method="cluster_bootstrap")

    point = statistic(flat)
    usable = [c for c in clusters if len(c) > 0]
    if len(usable) < 2:
        return Interval(point=point, lo=point, hi=point, level=level, method="cluster_bootstrap")

    rng = np.random.default_rng(seed)
    n = len(usable)
    draws = np.empty(n_resamples, dtype=float)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample: list[float] = []
        for i in idx:
            sample.extend(usable[i])
        draws[b] = statistic(sample) if sample else point

    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(draws, alpha))
    hi = float(np.quantile(draws, 1.0 - alpha))
    return Interval(point=float(point), lo=lo, hi=hi, level=level, method="cluster_bootstrap")


def bootstrap_mean(
    clusters: Sequence[Sequence[float]],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    return cluster_bootstrap(
        clusters,
        lambda xs: float(np.mean(xs)) if len(xs) else 0.0,
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )


def paired_cluster_bootstrap(
    pairs: Sequence[tuple[float, float]],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval on a difference where both measurements come from the same cluster.

    Used for the model-scale comparison, where the same entities are evaluated at two
    model sizes and an unpaired interval would be needlessly wide.
    """
    if not pairs:
        return Interval(0.0, 0.0, 0.0, level, "paired_cluster_bootstrap")
    diffs = [[a - b] for a, b in pairs]
    out = bootstrap_mean(diffs, n_resamples=n_resamples, level=level, seed=seed)
    return Interval(out.point, out.lo, out.hi, level, "paired_cluster_bootstrap")


def benjamini_hochberg(pvals: Mapping[str, float], q: float = 0.10) -> dict[str, float]:
    """BH adjusted p-values.

    BH controls the expected share of false findings among those called significant,
    which is the right target for exploratory per-family and per-feature breakdowns.
    Controlling family-wise error instead would be needlessly conservative given how
    many secondary quantities the report contains.
    """
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adjusted: dict[str, float] = {}
    prev = 1.0
    # walk from the largest p downward so the monotonicity step is a running minimum
    for rank in range(m, 0, -1):
        name, p = items[rank - 1]
        val = min(prev, p * m / rank)
        adjusted[name] = float(min(1.0, max(0.0, val)))
        prev = adjusted[name]
    return adjusted


def permutation_test_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    n_draws: int = 2_000,
    seed: int = 0,
) -> float:
    """Fraction of label permutations reaching an AUC at least as high as observed.

    Permutes labels at the group level, so the pair structure is preserved and the
    null is calibrated to the actual procedure rather than to a parametric
    approximation that would ignore clustering.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return float("nan")

    observed = auc_score(y.tolist(), s.tolist())
    rng = np.random.default_rng(seed)

    uniq = sorted(set(groups))
    idx_by_group = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}

    at_least = 0
    valid = 0
    for _ in range(n_draws):
        permuted = y.copy()
        for g in uniq:
            idx = idx_by_group[g]
            vals = y[idx]
            permuted[idx] = rng.permutation(vals)
        if len(set(permuted.tolist())) < 2:
            continue
        valid += 1
        if auc_score(permuted.tolist(), s.tolist()) >= observed:
            at_least += 1

    if valid == 0:
        return float("nan")
    # add-one keeps the p-value strictly positive, which is standard for a Monte
    # Carlo test and avoids reporting p = 0 from a finite number of draws
    return float((at_least + 1) / (valid + 1))


def power_note(n_pairs: int, threshold: float = 0.60) -> str:
    """Plain-language statement of what this sample size can and cannot resolve.

    Uses the Hanley and McNeil approximation for the standard error of AUC under the
    null. Printed next to any undetectability claim, because "we did not detect it"
    and "it is not there" are different statements and only the first is supported.
    """
    if n_pairs < 2:
        return "Too few pairs to say anything about detectability."
    se = math.sqrt((1.0 / 12.0) * (1.0 / n_pairs + 1.0 / n_pairs))
    resolvable = 0.5 + 1.96 * se
    return (
        f"With {n_pairs} matched pairs the standard error on AUC is about {se:.3f}, "
        f"so this study can distinguish 0.5 from roughly {resolvable:.2f} or higher. "
        f"An AUC below that is consistent with an undetectable barrier and also "
        f"consistent with a small effect this sample cannot resolve. "
        f"An undetectability claim here means the upper confidence bound sits at or "
        f"below {threshold:.2f}, not that no signal exists."
    )
