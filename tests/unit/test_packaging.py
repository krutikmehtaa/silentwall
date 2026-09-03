"""Packaging integrity.

These exist because of a real failure. The `.gitignore` had an unanchored `cache/`
pattern, which matched `src/silentwall/cache/` as well as the runtime cache directory
at the repo root. The whole subpackage was therefore never committed, every local check
passed, and CI failed on a fresh clone with ModuleNotFoundError for a module that
plainly existed on disk.

A missing subpackage is exactly the kind of fault that is invisible where you develop
and fatal everywhere else, so it gets a test rather than a note in the README.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import silentwall

#: Subpackages that must always be importable. Listed explicitly rather than discovered,
#: so that a subpackage vanishing from the distribution is a failure rather than a
#: shorter loop.
REQUIRED_SUBPACKAGES = (
    "silentwall.backends",
    "silentwall.cache",
    "silentwall.containment",
    "silentwall.corpus",
    "silentwall.probes",
    "silentwall.report",
    "silentwall.runner",
    "silentwall.scoring",
)

REQUIRED_MODULES = (
    "silentwall.cli",
    "silentwall.config",
    "silentwall.errors",
    "silentwall.hashing",
    "silentwall.pipeline",
    "silentwall.types",
)


@pytest.mark.parametrize("name", REQUIRED_SUBPACKAGES + REQUIRED_MODULES)
def test_required_module_is_importable(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_every_module_on_disk_imports() -> None:
    """Walk the installed package and import everything.

    Catches a module that is present but broken, and a subpackage that is absent from
    the distribution, in one pass. The GPU backend is skipped because importing it
    pulls in torch, which is deliberately not a core dependency.
    """
    skip = {"silentwall.backends.hf"}
    failures: list[str] = []

    for info in pkgutil.walk_packages(silentwall.__path__, prefix="silentwall."):
        if info.name in skip:
            continue
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 the point is to report, not to handle
            failures.append(f"{info.name}: {type(exc).__name__}: {exc}")

    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_no_source_file_is_git_ignored() -> None:
    """No file under src/ or tests/ may be excluded by .gitignore.

    This is the direct guard against the original bug. It reads .gitignore rather than
    shelling out to git, so it works in a source checkout without a git binary.
    """
    root = Path(__file__).resolve().parents[2]
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        pytest.skip("no .gitignore in this checkout")

    patterns = [
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # A directory pattern with no leading slash matches at any depth, which is what
    # caused the original fault. Flag any such pattern whose name collides with a real
    # directory under src/ or tests/.
    source_dirs = {
        p.name
        for base in ("src", "tests")
        for p in (root / base).rglob("*")
        if p.is_dir() and "__pycache__" not in p.parts
    }

    dangerous = [
        pat
        for pat in patterns
        if pat.endswith("/") and not pat.startswith("/") and pat.rstrip("/") in source_dirs
    ]

    assert not dangerous, (
        "these .gitignore patterns are unanchored and collide with real source "
        f"directories, so they would silently exclude code: {dangerous}. "
        f"Anchor them with a leading slash."
    )
