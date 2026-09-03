"""Property tests for cache keys, features, resumability and substitutability."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from silentwall.backends.base import GenerationRequest, derive_seed
from silentwall.backends.stub import StubBackend
from silentwall.cache.keys import cache_key
from silentwall.cache.store import DiskCacheLayer, LayeredCache, MemoryCacheLayer
from silentwall.config import SamplingConfig
from silentwall.errors import CacheCorruptError
from silentwall.runner.checkpoint import Checkpoint
from silentwall.scoring.features import FEATURE_NAMES, extract_features
from silentwall.types import Generation

from ..strategies import entities, generation_groups, probes

FAST = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _req(**kw: object) -> GenerationRequest:
    base = {
        "prompt": "what is the price",
        "sampling": SamplingConfig(k=4, max_new_tokens=32),
        "sample_index": 0,
        "probe_id": "p1",
        "entity_id": "e1",
        "tool_state_hash": "",
        "want_logprobs": True,
        "system_prompt": "",
        "context_docs": (),
        "tools_exposed": (),
    }
    base.update(kw)
    return GenerationRequest(**base)  # type: ignore[arg-type]


# Feature: silentwall, Property 20: For any generation request, re-deriving the cache key
# from the same components yields the same digest, changing any one of the backend
# fingerprint, containment fingerprint, prompt, tool state hash, sampling parameters,
# sample index or seed yields a different digest, and issuing the same logical request
# twice invokes the backend exactly once.
@FAST
@given(
    prompt=st.text(min_size=1, max_size=60),
    idx=st.integers(min_value=0, max_value=15),
    k=st.integers(min_value=1, max_value=8),
)
def test_cache_key_correctness(prompt: str, idx: int, k: int) -> None:
    sampling = SamplingConfig(k=k, max_new_tokens=32)
    req = _req(prompt=prompt, sample_index=idx, sampling=sampling)

    base = cache_key("bfp", "cfp", req)
    assert base == cache_key("bfp", "cfp", req), "same inputs give the same key"

    # every component must matter
    assert cache_key("other", "cfp", req) != base
    assert cache_key("bfp", "other", req) != base
    assert (
        cache_key("bfp", "cfp", _req(prompt=prompt + "x", sample_index=idx, sampling=sampling))
        != base
    )
    assert (
        cache_key("bfp", "cfp", _req(prompt=prompt, sample_index=idx + 1, sampling=sampling))
        != base
    )
    assert (
        cache_key(
            "bfp",
            "cfp",
            _req(prompt=prompt, sample_index=idx, sampling=sampling, tool_state_hash="t"),
        )
        != base
    )
    assert (
        cache_key(
            "bfp",
            "cfp",
            _req(prompt=prompt, sample_index=idx, sampling=replace(sampling, temperature=0.123)),
        )
        != base
    )
    assert (
        cache_key(
            "bfp",
            "cfp",
            _req(prompt=prompt, sample_index=idx, sampling=replace(sampling, max_new_tokens=999)),
        )
        != base
    )
    assert (
        cache_key(
            "bfp",
            "cfp",
            _req(prompt=prompt, sample_index=idx, sampling=replace(sampling, base_seed=7777)),
        )
        != base
    )
    assert (
        cache_key(
            "bfp",
            "cfp",
            _req(prompt=prompt, sample_index=idx, sampling=sampling, system_prompt="rule"),
        )
        != base
    )


@FAST
@given(idx=st.integers(min_value=0, max_value=20), base=st.integers(min_value=0, max_value=10**6))
def test_seed_is_deterministic_per_sample(idx: int, base: int) -> None:
    """Sample 0 must always be the same sample, otherwise leak@1 is arbitrary."""
    a = derive_seed(base, "prompt text", idx)
    b = derive_seed(base, "prompt text", idx)
    assert a == b
    assert derive_seed(base, "prompt text", idx + 1) != a
    assert derive_seed(base + 1, "prompt text", idx) != a
    assert 0 <= a < 2**63


def test_backend_called_once_per_logical_request() -> None:
    """Second ask for the same request must hit the cache, not the model."""
    calls = {"n": 0}
    inner = StubBackend()

    class Counting(StubBackend):
        def generate(self, requests):  # type: ignore[no-untyped-def]
            calls["n"] += len(requests)
            return inner.generate(requests)

    backend = Counting()
    cache = LayeredCache([MemoryCacheLayer()])
    req = _req()
    key = cache_key(backend.fingerprint(), "cfp", req)

    hit = cache.get(key)
    assert hit is None
    gen = backend.generate([req])[0]
    cache.put(Generation.from_dict({**gen.to_dict(), "cache_key": key}))
    assert calls["n"] == 1

    again = cache.get(key)
    assert again is not None
    assert calls["n"] == 1, "cached request must not reach the backend"


def test_cache_flags_stale_latency(tmp_path: Path) -> None:
    """A latency value from a previous session must not be presented as fresh.

    Latency is a behavioural feature. Serving a cached measurement as if it were
    measured now would fabricate a signal the detector could read.
    """
    layer = DiskCacheLayer(tmp_path / "c", writable=True)
    gen = Generation(
        cache_key="ab" + "0" * 14,
        probe_id="p",
        entity_id="e",
        sample_index=0,
        seed=1,
        text="hello world",
        n_tokens=2,
        backend_fp="b",
        containment_fp="c",
        latency_ms=42.0,
        latency_trustworthy=True,
    )
    layer.put(gen)

    fresh = DiskCacheLayer(tmp_path / "c", writable=True)
    got = fresh.get(gen.cache_key)
    assert got is not None
    assert got.latency_ms == 42.0
    assert got.latency_trustworthy is False


def test_cache_quarantines_mismatched_key(tmp_path: Path) -> None:
    layer = DiskCacheLayer(tmp_path / "c", writable=True)
    gen = Generation(
        cache_key="cd" + "0" * 14,
        probe_id="p",
        entity_id="e",
        sample_index=0,
        seed=1,
        text="x",
        n_tokens=1,
        backend_fp="b",
        containment_fp="c",
    )
    layer.put(gen)

    # Plant the record under a key it does not own. Lookup shards on the first two
    # characters of the requested key, so the corrupt entry has to live in shard "ff"
    # for a get("ff...") to find it.
    wrong_key = "ff" + "0" * 14
    layer._loaded_shards["ff"] = {wrong_key: gen}  # noqa: SLF001

    with pytest.raises(CacheCorruptError):
        layer.get(wrong_key)

    # and the layered cache treats that as a miss rather than propagating
    layered = LayeredCache([layer])
    assert layered.get(wrong_key) is None


def test_readonly_layer_refuses_writes(tmp_path: Path) -> None:
    ro = DiskCacheLayer(tmp_path / "ro", writable=False)
    gen = Generation(
        cache_key="ee" + "0" * 14,
        probe_id="p",
        entity_id="e",
        sample_index=0,
        seed=1,
        text="x",
        n_tokens=1,
        backend_fp="b",
        containment_fp="c",
    )
    with pytest.raises(CacheCorruptError):
        ro.put(gen)


def test_layered_cache_reads_through_and_writes_to_overlay(tmp_path: Path) -> None:
    """Prior sessions mount read only, the working directory takes the writes."""
    prior = DiskCacheLayer(tmp_path / "prior", writable=True)
    old = Generation(
        cache_key="aa" + "0" * 14,
        probe_id="p",
        entity_id="e",
        sample_index=0,
        seed=1,
        text="from prior session",
        n_tokens=3,
        backend_fp="b",
        containment_fp="c",
    )
    prior.put(old)

    layered = LayeredCache(
        [
            DiskCacheLayer(tmp_path / "prior", writable=False),
            DiskCacheLayer(tmp_path / "now", writable=True),
        ]
    )
    assert layered.get(old.cache_key) is not None

    new = Generation(
        cache_key="bb" + "0" * 14,
        probe_id="p",
        entity_id="e",
        sample_index=1,
        seed=2,
        text="this session",
        n_tokens=2,
        backend_fp="b",
        containment_fp="c",
    )
    layered.put(new)
    assert (tmp_path / "now").exists()
    assert DiskCacheLayer(tmp_path / "now", writable=False).get(new.cache_key) is not None
    assert DiskCacheLayer(tmp_path / "prior", writable=False).get(new.cache_key) is None


# Feature: silentwall, Property 21 (resumability half): a resumed run performs exactly
# the work not yet marked complete, and the union of interrupted and resumed output
# equals an uninterrupted run.
def test_checkpoint_resumes_without_repeating(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.txt"
    ck = Checkpoint(path)
    for uid in ("u1", "u2", "u3"):
        ck.mark(uid)

    reloaded = Checkpoint(path)
    assert reloaded.completed() == {"u1", "u2", "u3"}
    assert reloaded.is_done("u2")
    assert not reloaded.is_done("u4")

    # marking twice is a no-op, so a retried unit does not duplicate a line
    reloaded.mark("u1")
    assert len(Checkpoint(path).completed()) == 3


def test_checkpoint_survives_truncated_tail(tmp_path: Path) -> None:
    """A kill mid-write costs one unit, not the file."""
    path = tmp_path / "ckpt.txt"
    ck = Checkpoint(path)
    ck.mark("good1")
    ck.mark("good2")

    with path.open("a", encoding="utf-8") as fh:
        fh.write("partial_no_newline")

    recovered = Checkpoint(path)
    assert {"good1", "good2"}.issubset(recovered.completed())


# Feature: silentwall, Property 15: For any set of k generations for one probe, the
# extracted feature row has the same key set regardless of input, every rate-valued
# feature lies in the closed unit interval, every count-valued feature is non-negative,
# the row is unchanged by permuting the order of the k generations, semantic diversity is
# 0 when all samples are identical and positive when at least two differ, and features
# requiring logprobs are encoded as missing rather than zero when unavailable.
@FAST
@given(gens=generation_groups(min_k=1, max_k=6), probe=probes(), entity=entities())
def test_feature_row_well_formed(gens: list, probe: object, entity: object) -> None:
    row = extract_features(gens, probe, entity)  # type: ignore[arg-type]

    assert set(row.values) == set(FEATURE_NAMES), "key set must not depend on input"

    for name in ("refusal_rate", "toolcall_abstain_rate"):
        v = row.values[name]
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"

    for name in ("length_mean", "hedge_density", "toolcall_count", "toolcall_entropy", "length_cv"):
        v = row.values[name]
        if v is not None:
            assert v >= 0.0, f"{name} negative: {v}"

    sd = row.values["semantic_diversity"]
    assert sd is not None
    assert 0.0 <= sd <= 1.0

    # order invariance
    shuffled = list(reversed(gens))
    other = extract_features(shuffled, probe, entity)  # type: ignore[arg-type]
    for name in FEATURE_NAMES:
        a, b = row.values[name], other.values[name]
        if a is None or b is None:
            assert a is None and b is None, f"{name} availability changed with order"
        else:
            assert a == pytest.approx(b, abs=1e-9), f"{name} changed with order"


def test_semantic_diversity_boundaries() -> None:
    from silentwall.types import Entity, Probe, ProbeFamily

    probe = Probe("p", ProbeFamily.BEHAVIOURAL, "e", "b1", "prompt")
    entity = Entity("e", "Acme", "1", "control", "73", 1, "pair1")

    def gen(i: int, text: str) -> Generation:
        return Generation(
            cache_key=f"k{i}",
            probe_id="p",
            entity_id="e",
            sample_index=i,
            seed=i,
            text=text,
            n_tokens=max(1, len(text.split())),
            backend_fp="b",
            containment_fp="c",
        )

    same = [gen(i, "the same answer every time") for i in range(4)]
    assert extract_features(same, probe, entity).values["semantic_diversity"] == pytest.approx(0.0)

    differ = [gen(0, "alpha beta gamma"), gen(1, "completely unrelated wording here")]
    assert extract_features(differ, probe, entity).values["semantic_diversity"] > 0.0


def test_missing_logprobs_are_missing_not_zero() -> None:
    """Zero-filling would create a constant that separates the classes for free."""
    from silentwall.types import Entity, Probe, ProbeFamily

    probe = Probe("p", ProbeFamily.BEHAVIOURAL, "e", "b1", "prompt")
    entity = Entity("e", "Acme", "1", "control", "73", 1, "pair1")

    no_trace = [
        Generation(
            cache_key=f"k{i}",
            probe_id="p",
            entity_id="e",
            sample_index=i,
            seed=i,
            text="an answer",
            n_tokens=2,
            backend_fp="b",
            containment_fp="c",
            trace=None,
        )
        for i in range(3)
    ]
    row = extract_features(no_trace, probe, entity)
    assert row.values["token_entropy"] is None
    assert row.values["mean_neg_logprob"] is None
    assert row.values["confidence_gap"] is None
    # but the non-logprob features are still measured
    assert row.values["refusal_rate"] is not None
    assert row.values["semantic_diversity"] is not None


def test_empty_generation_group_returns_all_missing() -> None:
    from silentwall.types import Entity, Probe, ProbeFamily

    probe = Probe("p", ProbeFamily.BEHAVIOURAL, "e", "b1", "prompt")
    entity = Entity("e", "Acme", "1", "control", "73", 1, "pair1")
    row = extract_features([], probe, entity)
    assert set(row.values) == set(FEATURE_NAMES)
    assert all(v is None for v in row.values.values())
