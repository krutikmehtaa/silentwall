"""Checkpointing.

One completed unit id per line, flushed to disk as it happens. On restart the planner
recomputes the full unit list and subtracts what is already recorded. Losing at most
one unit to an unannounced session kill is the design target.

Append-only text is chosen over anything cleverer on purpose: a truncated final line
costs one unit, and the file stays readable when something goes wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["Checkpoint", "NullCheckpoint"]


class Checkpoint:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = self._load()

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        out: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                uid = line.strip()
                if uid:
                    out.add(uid)
        return out

    def completed(self) -> set[str]:
        return set(self._done)

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self._done

    def mark(self, unit_id: str) -> None:
        if unit_id in self._done:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(unit_id + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._done.add(unit_id)

    def reset(self) -> None:
        self._done.clear()
        if self.path.exists():
            self.path.unlink()

    def __len__(self) -> int:
        return len(self._done)


class NullCheckpoint:
    """No-op checkpoint for tests and in-memory runs."""

    def __init__(self) -> None:
        self._done: set[str] = set()

    def completed(self) -> set[str]:
        return set(self._done)

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self._done

    def mark(self, unit_id: str) -> None:
        self._done.add(unit_id)

    def reset(self) -> None:
        self._done.clear()

    def __len__(self) -> int:
        return len(self._done)
