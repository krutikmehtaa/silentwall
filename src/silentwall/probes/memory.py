"""Shared memory store for the memory-retrieval probe family.

This is the cross-audience channel. A scripted private-side phase writes notes about
a deal, then the public-side agent gets probed. If containment only covers the model
and the retrieval index, the memory store is where the information walks across the
barrier.

Backed by JSON so a run is inspectable after the fact, and so memory state can be
hashed into cache keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..hashing import hash_obj
from ..types import PrivateArtifact

__all__ = ["MemoryEntry", "MemoryStore", "prime_from_artifacts"]


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    entry_id: str
    entity_id: str
    text: str
    written_by: str
    is_private: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "entity_id": self.entity_id,
            "text": self.text,
            "written_by": self.written_by,
            "is_private": self.is_private,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> MemoryEntry:
        return cls(
            entry_id=str(d["entry_id"]),
            entity_id=str(d["entity_id"]),
            text=str(d["text"]),
            written_by=str(d["written_by"]),
            is_private=bool(d.get("is_private", False)),
        )


class MemoryStore:
    """Append-only shared memory, keyed by entity."""

    def __init__(self, entries: Iterable[MemoryEntry] = ()) -> None:
        self._entries: list[MemoryEntry] = list(entries)

    def write(
        self, entity_id: str, text: str, written_by: str, is_private: bool = False
    ) -> MemoryEntry:
        entry = MemoryEntry(
            entry_id=hash_obj("mem", entity_id, text, written_by)[:16],
            entity_id=entity_id,
            text=text,
            written_by=written_by,
            is_private=is_private,
        )
        self._entries.append(entry)
        return entry

    def read(self, entity_id: str) -> list[MemoryEntry]:
        return [e for e in self._entries if e.entity_id == entity_id]

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def fingerprint(self) -> str:
        return hash_obj([e.to_dict() for e in sorted(self._entries, key=lambda x: x.entry_id)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.to_dict() for e in self._entries]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MemoryStore:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(MemoryEntry.from_dict(d) for d in raw)


def prime_from_artifacts(
    store: MemoryStore,
    artifacts: Sequence[PrivateArtifact],
    deal_to_entity: dict[str, str],
) -> MemoryStore:
    """Simulate the private side taking notes.

    Deliberately writes a condensed version rather than the whole artifact, because
    that is what a real memory consolidation step would store, and because it makes
    the memory channel distinct from the retrieval channel rather than a duplicate
    of it.
    """
    for art in sorted(artifacts, key=lambda a: a.artifact_id):
        entity_id = deal_to_entity.get(art.deal_id)
        if entity_id is None:
            continue
        first_body = next(
            (ln for ln in art.text.splitlines() if ln.strip() and ln.endswith(".") is not False),
            "",
        )
        summary_lines = [ln for ln in art.text.splitlines() if "per share" in ln or "premium" in ln]
        summary = " ".join(summary_lines) or first_body
        if summary.strip():
            store.write(
                entity_id=entity_id,
                text=f"Note from {art.kind}: {summary.strip()}",
                written_by="private_side",
                is_private=True,
            )
    return store
