"""Stage 5: leak scoring, behavioural features, statistics, detector."""

from __future__ import annotations

from .detector import AdversarialDetector, DesignMatrix, build_matrix, fit_eval_detector
from .features import (
    FEATURE_NAMES,
    PRIMARY_FEATURES,
    aggregate_entity_features,
    extract_features,
    is_refusal,
)
from .leak import (
    aggregate_leak,
    leak_at_k,
    leak_curve,
    normalize_numeric,
    score_leak,
    text_contains_value,
)
from .stats import (
    auc_score,
    benjamini_hochberg,
    bootstrap_mean,
    cluster_bootstrap,
    paired_cluster_bootstrap,
    permutation_test_auc,
    power_note,
)

__all__ = [
    "score_leak",
    "leak_at_k",
    "leak_curve",
    "aggregate_leak",
    "normalize_numeric",
    "text_contains_value",
    "extract_features",
    "aggregate_entity_features",
    "is_refusal",
    "FEATURE_NAMES",
    "PRIMARY_FEATURES",
    "auc_score",
    "cluster_bootstrap",
    "bootstrap_mean",
    "paired_cluster_bootstrap",
    "benjamini_hochberg",
    "permutation_test_auc",
    "power_note",
    "fit_eval_detector",
    "build_matrix",
    "DesignMatrix",
    "AdversarialDetector",
]
