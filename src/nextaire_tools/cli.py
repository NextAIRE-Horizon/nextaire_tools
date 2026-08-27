"""Command-line interface for :mod:`nextaire_tools`.

Exposed as the ``nextaire_tools`` console script (see ``[project.scripts]`` in
``pyproject.toml``). Provides:

* ``nextaire_tools --version`` — print the installed version.
* ``nextaire_tools info`` — report which optional dependencies are available.
* ``nextaire_tools preprocess INPUT OUTPUT`` — run a (configurable) cleaning pipeline over a
  tabular file and write the result.

The CLI is intentionally thin; everything it does is available as a normal Python
API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from nextaire_tools import __version__

_OPTIONAL_DEPS = {
    "torch": "deep",
    "cdsapi": "extract",
    "xarray": "extract",
    "cfgrib": "extract",
    "geopandas": "geo",
    "osmnx": "geo",
    "prophet": "forecast",
    "holidays": "holidays",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextaire_tools",
        description="Air-quality time-series preprocessing, extraction, and modeling.",
    )
    parser.add_argument("--version", action="version", version=f"nextaire_tools {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("info", help="Show version and optional-dependency status.")

    pre = sub.add_parser("preprocess", help="Run a cleaning pipeline over a file.")
    pre.add_argument("input", type=Path, help="Input CSV/Excel/Parquet file.")
    pre.add_argument("output", type=Path, help="Output file (format inferred from suffix).")
    pre.add_argument(
        "--config",
        type=Path,
        default=None,
        help='JSON file: {"steps": [{"step": "MissingValueHandler", "params": {...}}, ...]}.',
    )
    pre.add_argument("--time-col", default=None, help="Timestamp column name.")
    pre.add_argument(
        "--set-time-index",
        action="store_true",
        help="Use the parsed timestamp column as a sorted DatetimeIndex.",
    )
    return parser


def _cmd_info() -> int:
    print(f"nextaire_tools {__version__}\n")
    print("Optional dependencies:")
    width = max(len(name) for name in _OPTIONAL_DEPS)
    for name, extra in sorted(_OPTIONAL_DEPS.items()):
        available = importlib.util.find_spec(name) is not None
        mark = "available" if available else f"missing  (pip install 'nextaire_tools[{extra}]')"
        status = "OK " if available else "-- "
        print(f"  [{status}] {name.ljust(width)}  {mark}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from nextaire_tools.io import load_table, save_table
    from nextaire_tools.preprocessing import (
        MissingValueHandler,
        OutlierHandler,
        Pipeline,
        TemporalFeatures,
    )

    df = load_table(
        args.input,
        time_col=args.time_col,
        set_time_index=args.set_time_index,
    )

    if args.config is not None:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        steps = config["steps"] if isinstance(config, dict) else config
        pipe = Pipeline.from_config(steps)
    else:
        pipe = Pipeline(
            [
                MissingValueHandler(strategy="interpolate", limit=3),
                OutlierHandler(method="iqr", strategy="clip"),
                TemporalFeatures(),
            ]
        )

    result = pipe.fit_transform(df)
    written = save_table(result, args.output)
    print(f"Wrote {len(result)} rows x {result.shape[1]} columns to {written}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info()
    if args.command == "preprocess":
        return _cmd_preprocess(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
