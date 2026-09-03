"""Assemble the audit result.

The limitations list is not boilerplate. At 60 pairs a null result means "we could not
detect it", not "it is not there", and a report that does not say so is overclaiming.
Every limitation printed here is derived from the actual run rather than written in
advance.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping, Sequence

from ..config import SilentwallConfig, config_hash
from ..hashing import short_hash
from ..types import (
    AuditResult,
    DetectabilityResult,
    Generation,
    Interval,
    LeakResult,
    RunRecord,
)

__all__ = ["build_run_record", "build_audit_result", "library_versions", "git_commit"]


def library_versions() -> dict[str, str]:
    """Versions of everything that could change a number."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "torch", "transformers"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = "not installed"
    return out


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_record(
    cfg: SilentwallConfig,
    corpus_hash: str,
    backend_fp: str,
    containment_fp: str,
    split_audit: Mapping[str, object],
    elapsed_seconds: float,
    generations_total: int,
    generations_from_cache: int,
    quarantined: int = 0,
    deferred: int = 0,
) -> RunRecord:
    cfg_hash = config_hash(cfg)
    return RunRecord(
        run_id=short_hash(cfg_hash, corpus_hash, containment_fp, backend_fp, elapsed_seconds),
        config_hash=cfg_hash,
        corpus_hash=corpus_hash,
        backend_fp=backend_fp,
        containment_fp=containment_fp,
        library_versions=library_versions(),
        seeds={
            "base_seed": cfg.sampling.base_seed,
            "split_seed": cfg.split.split_seed,
            "synthetic_seed": cfg.corpus.synthetic_seed,
        },
        dev_ids_hash=str(split_audit.get("dev_ids_hash", "")),
        eval_ids_hash=str(split_audit.get("eval_ids_hash", "")),
        splits_disjoint=bool(split_audit.get("splits_disjoint", False)),
        elapsed_seconds=elapsed_seconds,
        generations_total=generations_total,
        generations_from_cache=generations_from_cache,
        quarantined=quarantined,
        deferred=deferred,
        git_commit=git_commit(),
    )


def derive_limitations(
    cfg: SilentwallConfig,
    detectability: Sequence[DetectabilityResult],
    leak: Sequence[LeakResult],
    gens: Sequence[Generation],
    run: RunRecord,
) -> list[str]:
    """Limitations read off the run, not written in advance."""
    out: list[str] = []

    primary = next((d for d in detectability if d.detector_id == "logreg_primary"), None)
    if primary is not None:
        out.append(
            f"Detectability was estimated from {primary.n_pairs} matched pairs. {primary.power_note}"
        )
        if primary.undetectable_claim:
            out.append(
                "The undetectable claim here means the upper confidence bound sits at or below "
                f"{cfg.stats.undetectable_threshold:.2f}. It is not a proof of invisibility."
            )

    if gens:
        trusted = sum(1 for g in gens if g.latency_trustworthy)
        frac = trusted / len(gens)
        if frac < 1.0:
            out.append(
                f"Only {frac:.0%} of latency measurements came from fresh generations. "
                f"Latency is excluded from the primary feature set for this reason."
            )

    if not any(g.trace is not None for g in gens):
        out.append(
            "No token-level logprobs were available, so entropy and confidence features "
            "were imputed rather than measured."
        )

    if cfg.corpus.synthetic:
        out.append(
            "The corpus was generated offline rather than pulled from SEC EDGAR. Structure and "
            "ground truth are exact, but the language is synthetic, so absolute leak rates are "
            "not comparable to a live-corpus run."
        )

    if cfg.sampling.k < 8:
        out.append(
            f"k was {cfg.sampling.k}, which is low for leak@k. Low-probability leaks are "
            f"likely undercounted."
        )

    if run.deferred:
        out.append(f"{run.deferred} work units were deferred after repeated failures.")
    if run.quarantined:
        out.append(f"{run.quarantined} cache records were quarantined and regenerated.")

    return out


def build_audit_result(
    cfg: SilentwallConfig,
    method_id: str,
    run: RunRecord,
    leak: Sequence[LeakResult],
    detectability: Sequence[DetectabilityResult],
    utility: Mapping[str, Interval],
    gens: Sequence[Generation],
) -> AuditResult:
    return AuditResult(
        run=run,
        method_id=method_id,
        leak=tuple(leak),
        detectability=tuple(detectability),
        utility=dict(utility),
        limitations=tuple(derive_limitations(cfg, detectability, leak, gens, run)),
    )
