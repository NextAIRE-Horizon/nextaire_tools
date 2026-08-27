"""Exploratory data-analysis charts for air-quality time series.

Every function accepts a :class:`pandas.DataFrame`, resolves the columns it
needs with :mod:`nextaire_tools.utils.validation`, applies the shared nextaire_tools style, and
returns the matplotlib :class:`~matplotlib.axes.Axes` (or an array of axes for
small-multiple grids). Functions never mutate the caller's frame and never
call ``plt.show``.

The charts here answer the first questions asked of a new dataset: what is
missing, how are values distributed, how do variables correlate, how do they
evolve over time, and what seasonal structure they carry.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from pandas.api import types as pdt

from nextaire_tools._typing import ColumnLike, PathType
from nextaire_tools.exceptions import ColumnNotFoundError, ConfigurationError, SchemaError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import (
    check_dataframe,
    ensure_datetime_index,
    resolve_columns,
)
from nextaire_tools.viz.style import (
    DIVERGING_CMAP,
    INK,
    PALETTE,
    SEQUENTIAL_CMAP,
    _ensure_style,
    _finalize,
    _prepare_ax,
)

__all__ = [
    "plot_correlation",
    "plot_distributions",
    "plot_missingness",
    "plot_seasonality",
    "plot_timeseries",
]

_LOG = get_logger(__name__)


def _numeric_columns(df: pd.DataFrame, columns: ColumnLike | None) -> list[Hashable]:
    """Resolve ``columns`` and keep only the numeric ones.

    Raises
    ------
    SchemaError
        If no numeric columns remain after filtering.
    """
    resolved = resolve_columns(df, columns)
    numeric = [c for c in resolved if pdt.is_numeric_dtype(df[c])]
    if not numeric:
        raise SchemaError("No numeric columns available to plot.")
    return numeric


def plot_missingness(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    kind: str = "bar",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Visualize missing values per column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    columns : hashable or sequence of hashable, optional
        Columns to inspect. When ``None`` all columns are used.
    kind : {"bar", "matrix"}, default "bar"
        ``"bar"`` draws the missing fraction per column, sorted so the most
        incomplete column is on top. ``"matrix"`` draws a rows-by-columns map
        where missing cells are dark.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size when creating a new figure.
    save_path : str or os.PathLike, optional
        If given, save the figure (tight bbox, 150 dpi) and return.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.barh` (bar) or
        :meth:`matplotlib.axes.Axes.imshow` (matrix).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ConfigurationError
        If ``kind`` is not one of ``{"bar", "matrix"}``.

    Examples
    --------
    >>> ax = plot_missingness(df, kind="bar")  # doctest: +SKIP
    """
    if kind not in {"bar", "matrix"}:
        raise ConfigurationError(f"kind must be 'bar' or 'matrix', got {kind!r}.")

    frame = check_dataframe(df, copy=False)
    cols = resolve_columns(frame, columns)
    fig, ax = _prepare_ax(ax, figsize)

    if kind == "bar":
        frac = frame[cols].isna().mean().sort_values(ascending=True)
        positions = np.arange(len(frac))
        ax.barh(positions, frac.to_numpy(), color=PALETTE[0], **kwargs)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(c) for c in frac.index])
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Missing fraction")
        ax.set_title("Missingness by column")
        return _finalize(fig, ax, save_path)

    # kind == "matrix"
    mat = frame[cols].isna().to_numpy(dtype=float)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        **kwargs,
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
    ax.set_ylabel("Row index")
    ax.set_title("Missing-value map")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("missing", color=INK["secondary"])
    return _finalize(fig, ax, save_path, despine=False, grid=False)


def plot_distributions(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    bins: int = 40,
    kde: bool = True,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Draw a grid of histograms, one per numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    columns : hashable or sequence of hashable, optional
        Columns to plot. Non-numeric columns are dropped. When ``None`` all
        numeric columns are used.
    bins : int, default 40
        Number of histogram bins.
    kde : bool, default True
        Overlay a kernel-density estimate.
    ncols : int, default 3
        Number of columns in the subplot grid.
    figsize : tuple of float, optional
        Figure size; defaults to ``(4 * ncols, 3 * nrows)``.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :func:`seaborn.histplot`.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The grid of axes; unused cells (if any) are hidden.

    Examples
    --------
    >>> axes = plot_distributions(df, ncols=2)  # doctest: +SKIP
    """
    frame = check_dataframe(df, copy=False)
    cols = _numeric_columns(frame, columns)

    _ensure_style()
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(cols) / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    for i, col in enumerate(cols):
        sns.histplot(
            frame[col].dropna(),
            bins=bins,
            kde=kde,
            ax=flat[i],
            color=PALETTE[0],
            **kwargs,
        )
        flat[i].set_title(str(col))
        flat[i].set_xlabel("")

    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout()
    _finalize(fig, axes, save_path)
    return axes


def plot_correlation(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    method: str = "pearson",
    cluster: bool = False,
    annot: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes | sns.matrix.ClusterGrid:
    """Draw a correlation heatmap using the diverging (blue<->red) ramp.

    The colormap is centered at zero with ``vmin=-1`` and ``vmax=1`` so the
    neutral gray midpoint marks "no correlation".

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    columns : hashable or sequence of hashable, optional
        Numeric columns to correlate. When ``None`` all numeric columns.
    method : {"pearson", "kendall", "spearman"}, default "pearson"
        Correlation method passed to :meth:`pandas.DataFrame.corr`.
    cluster : bool, default False
        When ``True`` use :func:`seaborn.clustermap` (hierarchically ordered)
        and return its :class:`~seaborn.matrix.ClusterGrid`.
    annot : bool, default False
        Annotate each cell with its numeric value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (ignored when ``cluster=True``).
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :func:`seaborn.heatmap` (or :func:`seaborn.clustermap`).

    Returns
    -------
    matplotlib.axes.Axes or seaborn.matrix.ClusterGrid
        The axes for a plain heatmap, or the ClusterGrid when ``cluster``.

    Examples
    --------
    >>> ax = plot_correlation(df, method="spearman")  # doctest: +SKIP
    """
    frame = check_dataframe(df, copy=False)
    cols = _numeric_columns(frame, columns)
    corr = frame[cols].corr(method=method)

    if cluster:
        _ensure_style()
        grid = sns.clustermap(
            corr,
            cmap=DIVERGING_CMAP,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=annot,
            figsize=figsize,
            **kwargs,
        )
        if save_path is not None:
            grid.savefig(save_path, bbox_inches="tight", dpi=150)
        return grid

    fig, ax = _prepare_ax(ax, figsize)
    sns.heatmap(
        corr,
        cmap=DIVERGING_CMAP,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=annot,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        **kwargs,
    )
    ax.set_title(f"Correlation ({method})")
    return _finalize(fig, ax, save_path, despine=False, grid=False)


def plot_timeseries(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    time_col: Hashable | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot one line per numeric column against a datetime axis.

    Series are colored from the categorical palette in column order (color
    follows the entity, never its rank). A legend is drawn when two or more
    series are present; a single series is named by the title instead.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    columns : hashable or sequence of hashable, optional
        Numeric columns to plot. When ``None`` all numeric columns.
    time_col : hashable, optional
        Column holding timestamps. When ``None`` the existing (datetime)
        index is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = plot_timeseries(df, ["no2", "o3"], time_col="timestamp")  # doctest: +SKIP
    """
    frame = check_dataframe(df, copy=False)
    indexed = ensure_datetime_index(frame, time_col=time_col)
    cols = _numeric_columns(indexed, columns)

    if len(cols) > len(PALETTE):
        _LOG.warning(
            "plot_timeseries received %d series but the palette has %d colors; "
            "consider small multiples or aggregating minor series into 'Other'.",
            len(cols),
            len(PALETTE),
        )

    fig, ax = _prepare_ax(ax, figsize)
    for i, col in enumerate(cols):
        ax.plot(
            indexed.index,
            indexed[col].to_numpy(),
            label=str(col),
            color=PALETTE[i % len(PALETTE)],
            **kwargs,
        )

    ax.set_xlabel("Time")
    if len(cols) >= 2:
        ax.legend()
    else:
        ax.set_title(str(cols[0]))
    return _finalize(fig, ax, save_path)


def plot_seasonality(
    df: pd.DataFrame,
    column: Hashable,
    *,
    by: str = "hour",
    time_col: Hashable | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Box-plot a column grouped by a calendar component.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (never mutated).
    column : hashable
        Column to summarize.
    by : {"hour", "dayofweek", "month"}, default "hour"
        Calendar component to group by, taken from the datetime index.
    time_col : hashable, optional
        Column holding timestamps. When ``None`` the existing datetime index
        is used.
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
    ConfigurationError
        If ``by`` is not one of ``{"hour", "dayofweek", "month"}``.
    ColumnNotFoundError
        If ``column`` is absent from the (time-indexed) frame.

    Examples
    --------
    >>> ax = plot_seasonality(df, "no2", by="hour")  # doctest: +SKIP
    """
    if by not in {"hour", "dayofweek", "month"}:
        raise ConfigurationError(f"by must be one of 'hour', 'dayofweek', 'month', got {by!r}.")

    frame = check_dataframe(df, copy=False)
    indexed = ensure_datetime_index(frame, time_col=time_col)
    if column not in indexed.columns:
        raise ColumnNotFoundError(
            f"Column {column!r} not found. Available: {list(indexed.columns)}"
        )

    key = np.asarray(getattr(indexed.index, by))
    plot_df = pd.DataFrame({by: key, str(column): indexed[column].to_numpy()})

    fig, ax = _prepare_ax(ax, figsize)
    sns.boxplot(
        data=plot_df,
        x=by,
        y=str(column),
        ax=ax,
        color=PALETTE[0],
        fliersize=2.0,
        **kwargs,
    )
    ax.set_xlabel(by)
    ax.set_ylabel(str(column))
    ax.set_title(f"{column} by {by}")
    return _finalize(fig, ax, save_path)
