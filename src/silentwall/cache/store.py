"""On-disk generation store.

Durability is the requirement that shapes this file. Free-tier sessions terminate
without warning, so a generation is written and flushed before it is handed to the
caller. At most one in-flight unit is ever lost.

Layout is a sharded tree with one gzipped JSON Lines file per shard, plus a JSON
index for lookup. Sharding on the first two hex characters keeps any single file
small enough to append to cheaply, and JSON Lines means a truncated tail costs one
record rather than the whole shard.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..errors import CacheCorruptError
from ..types import Generation

__all__ = ["DiskCacheLayer", "MemoryCacheLayer"]


class MemoryCacheLayer:
    """In-process cache. Used by tests and as a hot layer in front of disk."""

    writable = True

    def __init__(self) -> None:
        self._store: dict[str, Generation] = {}

    def get(self, key: str) -> Generation | None:
        return self._store.get(key)

    def put(self, gen: Generation) -> None:
        self._store[gen.cache_key] = gen

    def keys(self) -> Iterator[str]:
        yield from self._store

    def __len__(self) -> int:
        return len(self._store)


class DiskCacheLayer:
    """Persistent shard-per-prefix store.

    Parameters
    ----------
    root:
        directory for this layer
    writable:
        False for layers mounted read only, which is how prior Kaggle sessions
        appear on the next run
    """

    def __init__(self, root: Path | str, writable: bool = True) -> None:
        self.root = Path(root)
        self.writable = writable
        self._index: dict[str, str] | None = None
        self._loaded_shards: dict[str, dict[str, Generation]] = {}
        self.quarantined = 0
        if writable:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "_quarantine").mkdir(exist_ok=True)

    # shard helpers

    def _shard_of(self, key: str) -> str:
        return key[:2]

    def _shard_path(self, shard: str) -> Path:
        return self.root / f"{shard}.jsonl.gz"

    def _load_shard(self, shard: str) -> dict[str, Generation]:
        if shard in self._loaded_shards:
            return self._loaded_shards[shard]

        path = self._shard_path(shard)
        found: dict[str, Generation] = {}
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            gen = Generation.from_dict(json.loads(line))
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            self.quarantined += 1
                            self._quarantine(line)
                            continue
                        found[gen.cache_key] = gen
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                # a shard truncated mid-write is recoverable: treat the whole shard as
                # a miss rather than losing the run
                self.quarantined += 1
                self._quarantine(f"unreadable shard {shard}: {exc}")

        self._loaded_shards[shard] = found
        return found

    def _quarantine(self, payload: str) -> None:
        if not self.writable:
            return
        qdir = self.root / "_quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        with (qdir / "bad_records.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(payload[:4000] + "\n")

    # public api

    def get(self, key: str) -> Generation | None:
        gen = self._load_shard(self._shard_of(key)).get(key)
        if gen is None:
            return None
        if gen.cache_key != key:
            self.quarantined += 1
            raise CacheCorruptError(f"record under {key} carries cache_key {gen.cache_key}")
        # A cached latency measurement was taken during a previous session, so it does
        # not describe this run. The feature extractor needs to know that, because
        # latency is one of the behavioural features and a stale value would be a
        # fabricated signal.
        return _mark_untrusted_latency(gen) if gen.latency_trustworthy else gen

    def put(self, gen: Generation) -> None:
        if not self.writable:
            raise CacheCorruptError(f"layer at {self.root} is read only")
        if not gen.cache_key:
            raise CacheCorruptError("cannot store a generation with an empty cache key")

        shard = self._shard_of(gen.cache_key)
        path = self._shard_path(shard)
        path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(gen.to_dict(), sort_keys=True, ensure_ascii=False)
        # append and flush before returning, so a kill after this point loses nothing
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        self._load_shard(shard)[gen.cache_key] = gen

    def count(self) -> int:
        total = 0
        for path in self.root.glob("*.jsonl.gz"):
            total += len(self._load_shard(path.name.split(".")[0]))
        return total


def _mark_untrusted_latency(gen: Generation) -> Generation:
    """Return a copy whose latency is flagged as coming from a previous session."""
    d = gen.to_dict()
    d["latency_trustworthy"] = False
    return Generation.from_dict(d)


class LayeredCache:
    """Ordered layers, read through all, write to the last writable one.

    This exists because of how free tiers persist data. On Kaggle the working
    directory lives for the session and can be published as a dataset, which then
    mounts read only on the next run. So prior sessions become earlier layers and the
    current working directory is the overlay. That is what makes a multi-session
    sweep additive instead of repetitive.
    """

    def __init__(self, layers: Sequence[DiskCacheLayer | MemoryCacheLayer]) -> None:
        if not layers:
            raise ValueError("LayeredCache needs at least one layer")
        self.layers = list(layers)
        writable = [layer for layer in self.layers if layer.writable]
        if not writable:
            raise ValueError("LayeredCache needs at least one writable layer")
        self.overlay = writable[-1]
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Generation | None:
        for layer in self.layers:
            try:
                gen = layer.get(key)
            except CacheCorruptError:
                continue
            if gen is not None:
                self.hits += 1
                return gen
        self.misses += 1
        return None

    def get_many(self, keys: Sequence[str]) -> dict[str, Generation]:
        out: dict[str, Generation] = {}
        for key in keys:
            gen = self.get(key)
            if gen is not None:
                out[key] = gen
        return out

    def put(self, gen: Generation) -> None:
        self.overlay.put(gen)

    def contains_many(self, keys: Sequence[str]) -> set[str]:
        """Which of these keys are already present, without counting hit stats."""
        present: set[str] = set()
        for key in keys:
            for layer in self.layers:
                try:
                    if layer.get(key) is not None:
                        present.add(key)
                        break
                except CacheCorruptError:
                    continue
        return present

    @property
    def quarantined(self) -> int:
        return sum(getattr(layer, "quarantined", 0) for layer in self.layers)
