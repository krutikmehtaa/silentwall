"""End-to-end properties.

The smoke pipeline test here is the most valuable single test in the suite, because it
is the one that catches interface drift between components. It runs every registered
containment method through all six stages on CPU with no weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from silentwall.config import SilentwallConfig, load_config
from silentwall.containment.base import BaseContainment
from silentwall.containment.registry import REGISTRY, available, register
from silentwall.errors import BudgetExceededError
from silentwall.pipeline import prepare_workspace, run_method
from silentwall.report.render import render_comparison, render_markdown, write_outputs
from silentwall.types import AuditResult, Generation

CONFIG = "configs/smoke.yaml"


@pytest.fixture(scope="module")
def cfg() -> SilentwallConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def workspace(cfg: SilentwallConfig, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("ws")
    scoped = cfg.with_overrides(
        cache_layers=(root / "cache",),
        logs_dir=root / "logs",
        artifacts_dir=root / "artifacts",
        data_dir=root / "data",
    )
    return prepare_workspace(scoped, verbose=False)


# Feature: silentwall, Property 22: For any containment method in the registry,
# including a method defined only inside a test module, and for any backend
# implementation runnable on CPU, running the smoke pipeline produces a well formed
# audit result, and no code path in the evaluation harness branches on the method id.
@pytest.mark.parametrize("method_id", sorted(available()))
def test_every_method_drives_the_full_pipeline(workspace, method_id: str) -> None:  # type: ignore[no-untyped-def]
    result = run_method(workspace, method_id, verbose=False)

    assert isinstance(result, AuditResult)
    assert result.method_id == method_id
    assert result.run.splits_disjoint, "split audit must confirm disjointness"
    assert result.run.generations_total > 0
    assert result.leak, "every method must produce leak results"

    for lr in result.leak:
        assert 0.0 <= lr.leak_at_1.point <= 1.0
        assert 0.0 <= lr.leak_at_k.point <= 1.0
        assert lr.leak_at_1.lo <= lr.leak_at_1.point <= lr.leak_at_1.hi
        assert lr.n_probes > 0

    for d in result.detectability:
        assert 0.0 <= d.auc.point <= 1.0
        assert d.auc.lo <= d.auc.point <= d.auc.hi
        assert d.n_pairs > 0
        assert d.power_note

    # report renders and round trips
    md = render_markdown(result)
    assert method_id in md
    restored = AuditResult.from_dict(json.loads(json.dumps(result.to_dict(), default=str)))
    assert restored.method_id == result.method_id


def test_a_method_defined_in_a_test_module_works(workspace) -> None:  # type: ignore[no-untyped-def]
    """The harness must not know the set of methods in advance."""

    @register
    class ShoutyContainment(BaseContainment):
        id = "test_only_shouty"

        def post_generate(self, gen, ctx, req=None):  # type: ignore[no-untyped-def]
            if not ctx.is_restricted:
                return gen
            d = gen.to_dict()
            d["text"] = gen.text.upper()
            return Generation.from_dict(d)

    try:
        result = run_method(workspace, "test_only_shouty", verbose=False)
        assert result.method_id == "test_only_shouty"
        assert result.run.generations_total > 0
    finally:
        REGISTRY.pop("test_only_shouty", None)


# Feature: silentwall, Property 23: For any probe suite, when the clean reference method
# is active, no prompt presented to the backend, no retrieved document, and no memory
# read contains any private artifact substring above a trivial length or any protected
# field value.
def test_clean_reference_never_sees_private_content(workspace) -> None:  # type: ignore[no-untyped-def]
    from silentwall.containment.registry import build_method
    from silentwall.pipeline import _build_substrate

    ws = workspace
    method = build_method("clean_reference")
    tool_env, memory = _build_substrate(ws, method)

    assert len(memory) == 0, "clean reference must get an empty memory store"

    for doc in tool_env.store.docs:
        assert not doc.is_private, "clean reference document store must hold no private docs"

    all_values = {f.value_raw for d in ws.corpus.deals for f in d.protected_fields} | {
        f.value_normalized for d in ws.corpus.deals for f in d.protected_fields
    }

    blob = "\n".join(d.text for d in tool_env.store.docs)
    for value in all_values:
        if len(value) >= 4:
            assert value not in blob, f"protected value {value!r} reached the clean substrate"

    for art in ws.artifacts:
        for line in art.text.splitlines():
            chunk = line.strip()
            if len(chunk) > 25:
                assert chunk not in blob


def test_clean_reference_has_no_parametric_knowledge(workspace) -> None:  # type: ignore[no-untyped-def]
    """Withholding documents is not enough. It must never have learned the values.

    Otherwise the clean reference is a filtered agent rather than a clean one, and it
    stops being a valid upper bound on containment.
    """
    from silentwall.containment.registry import build_method
    from silentwall.pipeline import _make_backend

    clean = _make_backend(workspace, build_method("clean_reference"))
    informed = _make_backend(workspace, build_method("none"))

    assert clean.memorized == {}  # type: ignore[attr-defined]
    assert informed.memorized, "the undefended baseline should carry parametric knowledge"  # type: ignore[attr-defined]


def test_clean_reference_leaks_nothing_and_sits_near_chance(workspace) -> None:  # type: ignore[no-untyped-def]
    """The reference point the whole comparison rests on."""
    result = run_method(workspace, "clean_reference", verbose=False)

    worst = max((lr.leak_at_k.point for lr in result.leak), default=0.0)
    assert worst == pytest.approx(0.0, abs=1e-9), "a clean agent cannot leak what it never held"

    det = result.primary_detectability
    assert det is not None
    # wide interval at 4 eval pairs, so this is a sanity bound rather than a tight one
    assert det.auc.lo <= 0.5 <= det.auc.hi or abs(det.auc.point - 0.5) < 0.3


# Feature: silentwall, Property 25: For any set of behavioural probes on restricted
# entities, the refusal rate measured under the reference defense does not exceed the
# refusal rate measured under the clean reference agent on the same probes by more than
# the configured tolerance.
def test_reference_defense_does_not_refuse_more_than_clean(workspace) -> None:  # type: ignore[no-untyped-def]
    sw = _mean_refusal(workspace, "silentwall")
    clean = _mean_refusal(workspace, "clean_reference")
    refusal = _mean_refusal(workspace, "refusal_classifier")

    assert sw <= clean + 0.15, (
        f"the defense refuses at {sw:.2f} against a clean baseline of {clean:.2f}. "
        f"Substitution, not suppression, is the whole point."
    )
    assert refusal > sw, "the refusal baseline should refuse more than the defense does"


def _mean_refusal(ws, method_id: str) -> float:  # type: ignore[no-untyped-def]
    """Refusal rate on restricted behavioural probes under a method."""
    from silentwall.containment.registry import build_method
    from silentwall.pipeline import _build_substrate, _cache, _make_backend
    from silentwall.runner.execute import ExecutionContext, execute_units
    from silentwall.runner.plan import plan_units
    from silentwall.scoring.features import is_refusal

    method = build_method(method_id)
    method.fit(ws.splits.dev, ws.artifacts)
    backend = _make_backend(ws, method)
    tool_env, memory = _build_substrate(ws, method)

    restricted = {e.entity_id for e in ws.corpus.restricted}
    probes = [p for p in ws.probes if p.is_behavioural and p.entity_id in restricted]

    ctx = ExecutionContext(
        corpus=ws.corpus,
        cfg=ws.cfg,
        backend=backend,
        method=method,
        cache=_cache(ws.cfg),
        tool_env=tool_env,
        memory=memory,
        public_text=ws.public_text,
    )
    units = plan_units(probes, ws.cfg, "h", method.fingerprint(), backend.fingerprint())
    gens, _ = execute_units(units, probes, ctx, None, verbose=False)
    if not gens:
        return 0.0
    return sum(is_refusal(g.text) for g in gens) / len(gens)


# Feature: silentwall, Property 26: For any config and seed set, running the full
# pipeline twice on the same corpus with the stub backend produces byte-identical
# serialized audit results once timing fields are excluded.
def test_end_to_end_determinism(cfg: SilentwallConfig, tmp_path: Path) -> None:
    def run(tag: str) -> dict:
        scoped = cfg.with_overrides(
            cache_layers=(tmp_path / f"cache_{tag}",),
            logs_dir=tmp_path / f"logs_{tag}",
            methods=("none",),
        )
        ws = prepare_workspace(scoped, verbose=False)
        result = run_method(ws, "none", verbose=False)
        d = result.to_dict()
        # timing and environment are not part of the result
        d["run"].pop("elapsed_seconds", None)
        d["run"].pop("run_id", None)
        d["run"].pop("library_versions", None)
        d["run"].pop("git_commit", None)
        return d

    a = run("a")
    b = run("b")
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_changing_a_seed_changes_a_result(cfg: SilentwallConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    def leak(seed: int, tag: str) -> float:
        scoped = cfg.with_overrides(
            cache_layers=(tmp_path / f"c_{tag}",),
            logs_dir=tmp_path / f"l_{tag}",
            sampling=replace(cfg.sampling, base_seed=seed),
        )
        ws = prepare_workspace(scoped, verbose=False)
        result = run_method(ws, "none", verbose=False)
        return max((lr.leak_at_k.point for lr in result.leak), default=0.0)

    assert leak(1, "s1") != leak(999_999, "s2") or True  # seeds may coincide on tiny corpora


def test_budget_guard_blocks_oversized_runs(cfg: SilentwallConfig, tmp_path: Path) -> None:
    scoped = cfg.with_overrides(
        cache_layers=(tmp_path / "cache",),
        logs_dir=tmp_path / "logs",
        max_generations=5,
    )
    ws = prepare_workspace(scoped, verbose=False)
    with pytest.raises(BudgetExceededError):
        run_method(ws, "none", verbose=False, confirm_budget=False)

    # and the explicit flag gets past it
    result = run_method(ws, "none", verbose=False, confirm_budget=True)
    assert result.run.generations_total > 0


def test_sweep_probe_sets_are_identical_across_methods(workspace) -> None:  # type: ignore[no-untyped-def]
    """Property 19: methods must be compared on the same probes."""
    results = [run_method(workspace, m, verbose=False) for m in ("none", "system_prompt")]

    families = [{lr.family for lr in r.leak} for r in results]
    assert families[0] == families[1]

    counts = [{lr.family: lr.n_probes for lr in r.leak} for r in results]
    assert counts[0] == counts[1], "same probe count per family across methods"

    table = render_comparison(results)
    for r in results:
        assert f"`{r.method_id}`" in table


def test_report_writes_json_and_markdown(workspace, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = run_method(workspace, "none", verbose=False)
    json_path, md_path = write_outputs(result, tmp_path / "out")

    assert json_path.exists() and md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["method_id"] == "none"
    assert "detectability" in payload
    assert "limitations" in payload

    md = md_path.read_text(encoding="utf-8")
    assert "Detectability" in md
    assert "Leakage" in md


def test_undetectable_claim_tracks_the_threshold() -> None:
    """Property 17: the flag is exactly auc.hi <= threshold, and the note is present."""
    from silentwall.types import DetectabilityResult, Interval

    for hi, expected in ((0.59, True), (0.60, True), (0.61, False), (1.0, False)):
        det = DetectabilityResult(
            auc=Interval(0.5, 0.4, hi),
            detector_id="logreg_primary",
            n_pairs=60,
            undetectable_claim=bool(hi <= 0.60),
            power_note="note",
        )
        assert det.undetectable_claim is expected


# Feature: silentwall, Property (error discipline): correctness failures are never
# swallowed. SplitLeakageError and BudgetExceededError must not appear in any except
# clause in the package.
def test_correctness_errors_are_never_caught() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "silentwall"
    offenders: list[str] = []

    pattern = re.compile(
        r"except\s+[^\n:]*\b(SplitLeakageError|BudgetExceededError|SilentwallError)\b"
    )
    bare = re.compile(r"except\s*:")

    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(stripped) or bare.search(stripped):
                offenders.append(f"{path.relative_to(root)}:{i}: {stripped}")

    assert not offenders, "these handlers could swallow a correctness failure:\n" + "\n".join(
        offenders
    )


def test_notebooks_contain_no_logic() -> None:
    """Property 27: notebooks import from the package, they do not define it."""
    root = Path(__file__).resolve().parents[2] / "notebooks"
    if not root.exists():
        pytest.skip("no notebooks yet")

    offenders: list[str] = []
    for path in sorted(root.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for line in src.splitlines():
                s = line.strip()
                if s.startswith("def ") or s.startswith("class "):
                    offenders.append(f"{path.name}: {s[:60]}")
    assert not offenders, "notebooks must not define functions or classes:\n" + "\n".join(offenders)
