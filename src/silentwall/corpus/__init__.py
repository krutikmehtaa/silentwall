"""Stage 0 and stage 1: corpus, artifacts, and split assignment."""

from __future__ import annotations

from .build import build_artifacts, build_corpus, build_synthetic_corpus, corpus_content_hash
from .controls import Candidate, assign_size_bands, build_entities, match_controls
from .parse import normalize_money, parse_filing, parse_or_exclude, render_filing
from .splits import DevSplit, EvalSplit, SplitAssignment, assign_splits
from .synth import TEMPLATE_PACK_VERSION, synthesize_artifacts, synthesize_public_text

__all__ = [
    "build_corpus",
    "build_synthetic_corpus",
    "build_artifacts",
    "corpus_content_hash",
    "Candidate",
    "assign_size_bands",
    "match_controls",
    "build_entities",
    "parse_filing",
    "parse_or_exclude",
    "render_filing",
    "normalize_money",
    "assign_splits",
    "SplitAssignment",
    "DevSplit",
    "EvalSplit",
    "synthesize_artifacts",
    "synthesize_public_text",
    "TEMPLATE_PACK_VERSION",
]
