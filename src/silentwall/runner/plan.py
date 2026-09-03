"""Work planning and the budget guard.

Planning runs before any model call, counts what is already cached, and prints the
projected cost. That ordering is the point: discovering a quota overrun three hours
into a free-tier session is the expensive failure mode, and it is entirely avoidable
by counting first.

A work unit is one probe on one entity with all k samples. That granularity sets how
much a killed session can cost, which is at most one unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import SilentwallConfig
from ..errors import BudgetExceededError
from ..hashing import hash_obj
from ..types import Probe

__all__ = ["WorkUnit", "BudgetEstimate", "plan_units", "check_budget", "format_budget"]

#: Throughput guesses per tier, completions per second, used only for the projection.
#: Deliberately conservative so the printed estimate is a ceiling.
_THROUGHPUT: dict[str, float] = {
    "stub": 4000.0,
    "cpu-0p5b": 1.2,
    "gpu-1p5b": 12.0,
    "gpu-8b-nf4": 4.0,
}


@dataclass(frozen=True, slots=True)
class WorkUnit:
    probe_id: str
    entity_id: str
    k: int
    unit_id: str
    family: str = ""

    @property
    def generations(self) -> int:
        return self.k


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    prompts: int
    generations: int
    cached: int
    to_generate: int
    projected_seconds: float
    tier: str

    @property
    def projected_hours(self) -> float:
        return self.projected_seconds / 3600.0


def make_unit_id(
    probe_id: str, entity_id: str, k: int, config_hash: str, containment_fp: str, backend_fp: str
) -> str:
    """Unit identity. Changing any fingerprint invalidates the checkpoint, which is
    correct: the same probe under a different method is different work."""
    return hash_obj(probe_id, entity_id, k, config_hash, containment_fp, backend_fp)[:20]


def plan_units(
    probes: Sequence[Probe],
    cfg: SilentwallConfig,
    config_hash: str,
    containment_fp: str,
    backend_fp: str,
) -> list[WorkUnit]:
    """One unit per probe, in a stable order."""
    k = cfg.sampling.k
    units = [
        WorkUnit(
            probe_id=p.probe_id,
            entity_id=p.entity_id,
            k=k,
            unit_id=make_unit_id(
                p.probe_id, p.entity_id, k, config_hash, containment_fp, backend_fp
            ),
            family=p.family.value,
        )
        for p in probes
    ]
    units.sort(key=lambda u: (u.entity_id, u.family, u.probe_id))
    return units


def estimate(
    units: Sequence[WorkUnit],
    cached_keys: int,
    tier: str,
) -> BudgetEstimate:
    total = sum(u.generations for u in units)
    to_generate = max(0, total - cached_keys)
    rate = _THROUGHPUT.get(tier, 3.0)
    return BudgetEstimate(
        prompts=len(units),
        generations=total,
        cached=cached_keys,
        to_generate=to_generate,
        projected_seconds=to_generate / rate if rate > 0 else 0.0,
        tier=tier,
    )


def format_budget(est: BudgetEstimate) -> str:
    """Human-readable cost table, printed before any work happens."""
    pct = (100.0 * est.cached / est.generations) if est.generations else 0.0
    if est.projected_seconds < 90:
        eta = f"{est.projected_seconds:.0f} sec"
    elif est.projected_seconds < 5400:
        eta = f"{est.projected_seconds / 60:.0f} min"
    else:
        eta = f"{est.projected_hours:.1f} hours"

    return "\n".join(
        [
            f"tier            {est.tier}",
            f"prompts         {est.prompts:,}",
            f"generations     {est.generations:,}",
            f"already cached  {est.cached:,} ({pct:.0f}%)",
            f"to generate     {est.to_generate:,}",
            f"projected time  {eta}",
        ]
    )


def check_budget(est: BudgetEstimate, cfg: SilentwallConfig, confirmed: bool = False) -> None:
    """Refuse to start an oversized run unless the caller says so explicitly."""
    if est.to_generate > cfg.max_generations and not confirmed:
        raise BudgetExceededError(
            f"this run needs {est.to_generate:,} generations, the configured ceiling is "
            f"{cfg.max_generations:,}. Projected time {est.projected_hours:.1f} hours. "
            f"Raise max_generations in the config, or pass confirm_budget to proceed."
        )
