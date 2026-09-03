"""Stage 6: reporting."""

from __future__ import annotations

from .build import build_audit_result, build_run_record, git_commit, library_versions
from .render import (
    adjust_secondary,
    render_comparison,
    render_markdown,
    write_comparison,
    write_outputs,
)

__all__ = [
    "build_run_record",
    "build_audit_result",
    "library_versions",
    "git_commit",
    "render_markdown",
    "render_comparison",
    "write_outputs",
    "write_comparison",
    "adjust_secondary",
]
