"""Outlier-inspection charts.

Two views: a set of horizontal box-plots for a quick multi-column overview,
and a single-series view that highlights out-of-bounds points and draws the
bound lines. Bounds may be passed explicitly or read (by duck typing) from a
fitted outlier-handler object exposing a ``bounds_`` mapping, so this module
never imports from :mod:`nextaire_tools.preprocessing` and stays dependency-light.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from pandas.api import types as pdt

from nextaire_tools._typing import ColumnLike, PathType
from nextaire_tools.exceptions import ColumnNotFoundError, ConfigurationError, SchemaError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import check_dataframe, resolve_columns
from nextaire_tools.viz.style import INK, PALETTE, _finalize, _prepare_ax

__all__ = [
    "plot_boxplots",
    "plot_outliers",
]

_LOG = get_logger(__name__)

#: Reserved status color for flagged (out-of-bounds) points.
_STATUS_RED = "#d03b3b"


def plot_boxplots(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw horizontal box-plots for the numeric columns of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    columns : hashable or sequence of hashable, optional
        Columns to plot. Non-numeric columns are dropped. When ``None`` all
        numeric columns are used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :func:`seaborn.boxplot`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    SchemaError
        If no numeric columns are available.

    Examples
    --------
    >>> ax = plot_boxplots(df, ["no2", "o3", "pm25"])  # doctest: +SKIP
    """
    frame = check_dataframe(df, copy=False)
    resolved = resolve_columns(frame, columns)
    numeric = [c for c in resolved if pdt.is_numeric_dtype(frame[c])]
    if not numeric:
        raise SchemaError("No numeric columns available to plot.")

    fig, ax = _prepare_ax(ax, figsize)
    sns.boxplot(
        data=frame[numeric],
        orient="h",
        color=PALETTE[0],
        fliersize=2.0,
        ax=ax,
        **kwargs,
    )
    ax.set_xlabel("Value")
    ax.set_title("Box-plots")
    return _finalize(fig, ax, save_path)


def _resolve_bounds(
    bounds: tuple[float | None, float | None] | None,
    handler: Any,
    column: Hashable,
) -> tuple[float | None, float | None]:
    """Resolve ``(low, high)`` from explicit bounds or a fitted handler.

    ``handler`` is duck-typed: any object exposing a ``bounds_`` mapping keyed
    by column label is accepted, where each value is a ``(low, high)`` pair or
    a mapping with ``"lower"``/``"upper"`` (or ``"low"``/``"high"``) keys.
    """
    if bounds is not None:
        low, high = bounds
        return low, high

    if handler is None:
        return None, None

    handler_bounds = getattr(handler, "bounds_", None)
    if not isinstance(handler_bounds, Mapping):
        raise ConfigurationError(
            "handler must expose a fitted 'bounds_' mapping (e.g. a fitted OutlierHandler)."
        )
    if column not in handler_bounds:
        raise ConfigurationError(
            f"Column {column!r} not present in handler.bounds_ (keys: {list(handler_bounds)})."
        )

    raw = handler_bounds[column]
    if isinstance(raw, Mapping):
        low = raw.get("lower", raw.get("low"))
        high = raw.get("upper", raw.get("high"))
        return low, high
    return raw[0], raw[1]


def plot_outliers(
    df: pd.DataFrame,
    column: Hashable,
    *,
    bounds: tuple[float | None, float | None] | None = None,
    handler: Any = None,
    time_col: Hashable | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a single series and highlight out-of-bounds points.

    Points outside ``[low, high]`` are drawn in the reserved status red and
    the bound lines are drawn as dashed horizontals. The x-axis is time when a
    datetime index (or ``time_col``) is available, otherwise a positional
    index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    column : hashable
        Column to inspect.
    bounds : tuple of (float or None, float or None), optional
        Explicit ``(low, high)`` bounds. Either end may be ``None`` (one-sided).
        Takes precedence over ``handler``.
    handler : object, optional
        A fitted outlier handler exposing a ``bounds_`` mapping keyed by
        column. Read via duck typing; never imported.
    time_col : hashable, optional
        Column holding timestamps for the x-axis. When ``None`` a datetime
        index is used if present.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the base series.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ColumnNotFoundError
        If ``column`` (or ``time_col``) is missing.
    ConfigurationError
        If ``handler`` is given but lacks a usable ``bounds_`` mapping.

    Examples
    --------
    >>> ax = plot_outliers(df, "no2", bounds=(0, 200))  # doctest: +SKIP
    >>> ax = plot_outliers(df, "no2", handler=fitted_handler)  # doctest: +SKIP
    """
    frame = check_dataframe(df, copy=False)
    if column not in frame.columns:
        raise ColumnNotFoundError(f"Column {column!r} not found. Available: {list(frame.columns)}")

    y = frame[column].to_numpy(dtype=float)

    if time_col is not None:
        if time_col not in frame.columns:
            raise ColumnNotFoundError(
                f"time_col={time_col!r} not found. Available: {list(frame.columns)}"
            )
        x: np.ndarray = pd.to_datetime(frame[time_col], errors="coerce").to_numpy()
        xlabel = "Time"
    elif isinstance(frame.index, pd.DatetimeIndex):
        x = frame.index.to_numpy()
        xlabel = "Time"
    else:
        x = np.arange(len(y))
        xlabel = "Index"

    low, high = _resolve_bounds(bounds, handler, column)

    fig, ax = _prepare_ax(ax, figsize)
    ax.plot(x, y, color=PALETTE[0], label=str(column), **kwargs)

    finite = np.isfinite(y)
    mask = np.zeros_like(y, dtype=bool)
    if low is not None:
        mask |= finite & (y < low)
    if high is not None:
        mask |= finite & (y > high)

    n_flagged = int(mask.sum())
    if n_flagged:
        ax.scatter(
            np.asarray(x)[mask],
            y[mask],
            color=_STATUS_RED,
            s=22,
            zorder=3,
            label=f"outlier (n={n_flagged})",
        )
    _LOG.debug("plot_outliers flagged %d of %d points for %r", n_flagged, len(y), column)

    for value in (low, high):
        if value is not None:
            ax.axhline(value, color=INK["baseline"], linestyle="--", linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(str(column))
    ax.set_title(str(column))
    if n_flagged:
        ax.legend()
    return _finalize(fig, ax, save_path)
