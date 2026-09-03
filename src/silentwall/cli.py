"""Command line interface.

Subcommands map onto pipeline stages so a session can do one stage, die, and be
resumed without redoing the previous ones. Everything takes a config path plus
optional dotted overrides, so nothing needs editing to change a parameter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SilentwallConfig, load_config
from .containment.registry import available
from .errors import (
    BackendOOMError,
    ConfigError,
    CorpusFetchError,
    MatchingInfeasibleError,
    ParseIncompleteError,
)
from .pipeline import Workspace, prepare_workspace, run_method, run_sweep, save_workspace
from .report.render import render_comparison, write_comparison, write_outputs

__all__ = ["main", "build_parser"]

#: Errors worth turning into a readable message. Correctness failures are excluded on
#: purpose, see the handler in main().
PRESENTABLE = (
    ConfigError,
    CorpusFetchError,
    MatchingInfeasibleError,
    ParseIncompleteError,
    BackendOOMError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silentwall",
        description="Audit whether an information barrier in an LLM agent is detectable",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", default="configs/smoke.yaml", help="path to a config file")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="dotted override, for example --set sampling.k=4",
        )
        p.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")

    p_corpus = sub.add_parser("corpus", help="stage 0 and 1, build corpus and probes, CPU only")
    common(p_corpus)
    p_corpus.add_argument("--out", default=None, help="where to write corpus artifacts")

    p_audit = sub.add_parser("audit", help="run one containment method end to end")
    common(p_audit)
    p_audit.add_argument("--method", "-m", required=True, help=f"one of {', '.join(available())}")
    p_audit.add_argument("--out", default="outputs", help="where to write the report")
    p_audit.add_argument(
        "--confirm-budget",
        action="store_true",
        help="proceed even if the projected generation count exceeds the configured ceiling",
    )

    p_sweep = sub.add_parser("sweep", help="run every method in the config and compare")
    common(p_sweep)
    p_sweep.add_argument("--out", default="outputs", help="where to write reports")
    p_sweep.add_argument("--confirm-budget", action="store_true")

    p_plan = sub.add_parser("plan", help="print the projected cost without generating anything")
    common(p_plan)

    sub.add_parser("methods", help="list registered containment methods")

    return parser


def _overrides(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"override {item!r} must look like key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "methods":
        print("registered containment methods:")
        for name in available():
            print(f"  {name}")
        return 0

    try:
        cfg = load_config(args.config, _overrides(args.set))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    verbose = not args.quiet

    try:
        if args.command == "corpus":
            ws = prepare_workspace(cfg, verbose=verbose)
            out = Path(args.out) if args.out else cfg.artifacts_dir / "corpus"
            path = save_workspace(ws, out)
            print(f"wrote corpus artifacts to {path}")
            return 0

        if args.command == "plan":
            ws = prepare_workspace(cfg, verbose=verbose)
            _print_plan(ws, cfg)
            return 0

        if args.command == "audit":
            ws = prepare_workspace(cfg, verbose=verbose)
            result = run_method(
                ws, args.method, verbose=verbose, confirm_budget=args.confirm_budget
            )
            json_path, md_path = write_outputs(result, args.out)
            print(f"\nwrote {md_path}")
            print(f"wrote {json_path}")
            return 0

        if args.command == "sweep":
            results = run_sweep(cfg, verbose=verbose, confirm_budget=args.confirm_budget)
            for r in results:
                write_outputs(r, args.out)
            json_path, md_path = write_comparison(results, args.out)
            print()
            print(render_comparison(results))
            print(f"wrote {md_path}")
            print(f"wrote {json_path}")
            return 0

    # Only errors that represent a recoverable situation for the user get a friendly
    # message. Correctness failures, meaning SplitLeakageError and BudgetExceededError,
    # are deliberately absent from this tuple so they propagate with a traceback. A
    # contaminated split or a silently truncated run must not look like a normal error
    # path, because the numbers it produces would still look plausible.
    except PRESENTABLE as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted. Progress is checkpointed, rerun to resume.", file=sys.stderr)
        return 130

    parser.print_help()
    return 1


def _print_plan(ws: Workspace, cfg: SilentwallConfig) -> None:
    """Cost projection per method, before anything expensive happens."""
    from .config import config_hash
    from .containment.registry import build_method
    from .runner.plan import estimate, format_budget, plan_units

    print("\nprojected cost per method")
    total = 0
    for method_id in cfg.methods:
        method = build_method(method_id)
        units = plan_units(ws.probes, cfg, config_hash(cfg), method.fingerprint(), "planning")
        est = estimate(units, 0, cfg.tier)
        total += est.to_generate
        print(f"\n{method_id}")
        for line in format_budget(est).splitlines():
            print(f"  {line}")

    print(f"\nsweep total: {total:,} generations across {len(cfg.methods)} methods")
    if cfg.tier != "stub":
        print("Nothing has been generated. Run sweep when the numbers above look acceptable.")


if __name__ == "__main__":
    raise SystemExit(main())
