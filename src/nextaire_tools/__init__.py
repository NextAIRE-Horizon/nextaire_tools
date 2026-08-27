"""nextaire_tools — air-quality time-series preprocessing, extraction, visualization, and ML/DL.

The most-used names are available directly on the top-level namespace::

    import nextaire_tools
    df = nextaire_tools.load_table("station.csv", time_col="timestamp", set_time_index=True)
    pipe = nextaire_tools.Pipeline(
        [nextaire_tools.MissingValueHandler(), nextaire_tools.TemporalFeatures()]
    )

Heavier subpackages (:mod:`nextaire_tools.viz`, :mod:`nextaire_tools.models`,
:mod:`nextaire_tools.extractors`) are imported lazily on first attribute access, so
``import nextaire_tools`` stays fast and does not require optional dependencies.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

try:
    __version__ = _pkg_version("nextaire_tools")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.1.0"

# --- Eager, lightweight core ------------------------------------------------
from nextaire_tools.exceptions import NextaireToolsError
from nextaire_tools.io import load_table, save_table
from nextaire_tools.preprocessing import (
    BaseStep,
    CorrelationFilter,
    LagFeatures,
    MissingValueHandler,
    OutlierHandler,
    Pipeline,
    Scaler,
    TemporalFeatures,
    WindDecomposer,
    make_pipeline,
)
from nextaire_tools.utils.logging import enable_logging, get_logger

# --- Lazy subpackages (avoid importing matplotlib/sklearn extras eagerly) ---
_LAZY_SUBMODULES = frozenset({"viz", "models", "extractors"})

if TYPE_CHECKING:  # for type checkers / IDEs only
    from nextaire_tools import extractors, models, viz


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"nextaire_tools.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'nextaire_tools' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__ = [
    "__version__",
    # core
    "load_table",
    "save_table",
    "Pipeline",
    "make_pipeline",
    "BaseStep",
    "MissingValueHandler",
    "OutlierHandler",
    "TemporalFeatures",
    "WindDecomposer",
    "LagFeatures",
    "CorrelationFilter",
    "Scaler",
    "enable_logging",
    "get_logger",
    "NextaireToolsError",
    # lazy subpackages
    "viz",
    "models",
    "extractors",
]
