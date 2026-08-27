"""Shared visual style for every :mod:`nextaire_tools` chart.

This module is the single source of truth for the nextaire_tools look: a validated,
colorblind-safe categorical palette, a single-hue sequential ramp, a
blue<->red diverging ramp with a neutral midpoint, and a small set of "ink"
tokens for text, axes, grid, and surface colors.

Every plotting function in :mod:`nextaire_tools.viz` applies this style via
:func:`set_style` (or the lower-level :func:`_prepare_ax` / :func:`_finalize`
helpers) so that all charts read as one system: thin marks, a recessive
hairline grid, hidden top/right spines, and a light off-white surface.

Notes
-----
Importing this module has no side effects beyond building two matplotlib
colormap objects. The global matplotlib ``rcParams`` are only touched when
:func:`set_style` (or :func:`_ensure_style`) is called, which happens lazily
the first time a nextaire_tools plotting function runs.

Examples
--------
>>> from nextaire_tools.viz.style import set_style, PALETTE
>>> set_style("talk")
>>> PALETTE[0]
'#2a78d6'
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure

from nextaire_tools._typing import PathType

__all__ = [
    "DIVERGING_CMAP",
    "INK",
    "PALETTE",
    "SEQUENTIAL_CMAP",
    "nextaire_tools_style",
    "set_style",
]

#: Categorical color cycle (light mode) in fixed order. Hues are assigned to
#: entities by position, never by rank, and never cycled beyond eight series.
PALETTE: list[str] = [
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
]

#: Non-series "ink" tokens: text, axis, grid, spine, and surface colors.
INK: dict[str, str] = {
    "primary": "#0b0b0b",  # primary text
    "secondary": "#52514e",  # secondary text / axis labels
    "muted": "#898781",  # muted axis ticks
    "grid": "#e1e0d9",  # hairline gridlines
    "baseline": "#c3c2b7",  # baseline / spines / reference lines
    "surface": "#fcfcfb",  # figure & axes background
}

#: Sequential (magnitude) ramp: a single blue hue, light -> dark.
SEQUENTIAL_CMAP: Colormap = LinearSegmentedColormap.from_list(
    "nextaire_tools_sequential", ["#cde2fb", "#0d366b"]
)

#: Diverging (polarity) ramp: blue <-> red with a neutral gray midpoint.
DIVERGING_CMAP: Colormap = LinearSegmentedColormap.from_list(
    "nextaire_tools_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)

# Tracks whether the global style has been applied at least once, so plotting
# helpers can apply it lazily without clobbering a user's explicit set_style.
_STYLE_APPLIED: bool = False


def set_style(context: str = "notebook", *, grid: bool = True) -> None:
    """Apply the nextaire_tools matplotlib/seaborn style globally.

    Sets the seaborn context and updates :data:`matplotlib.rcParams` with the
    nextaire_tools color tokens, the categorical color cycle, thin marks, hidden
    top/right spines, and a recessive hairline grid. The function is
    idempotent: calling it repeatedly leaves the style in the same state.

    Parameters
    ----------
    context : str, default "notebook"
        Seaborn plotting context ("paper", "notebook", "talk", "poster").
        Controls the base scaling of fonts and line widths.
    grid : bool, default True
        When ``True`` a light background grid is drawn on new axes.

    Returns
    -------
    None

    Examples
    --------
    >>> set_style("talk", grid=False)
    """
    global _STYLE_APPLIED

    sns.set_context(context)
    mpl.rcParams.update(
        {
            # Surfaces
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "savefig.edgecolor": INK["surface"],
            # Text / ink
            "text.color": INK["primary"],
            "axes.titlecolor": INK["primary"],
            "axes.labelcolor": INK["secondary"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            # Spines
            "axes.edgecolor": INK["baseline"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "axes.grid": grid,
            "axes.grid.axis": "both",
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            # Marks
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.0,
            # Legend
            "legend.frameon": False,
            # Fonts
            "font.family": "sans-serif",
            # Categorical cycle assigned by entity, in fixed order.
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )
    _STYLE_APPLIED = True


def _ensure_style() -> None:
    """Apply :func:`set_style` once if it has not been applied yet.

    Lets plotting helpers guarantee the nextaire_tools look without overriding a style
    the caller may have set explicitly (e.g. a different seaborn context).
    """
    if not _STYLE_APPLIED:
        set_style()


def _prepare_ax(
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Return a ``(fig, ax)`` pair, creating them if ``ax`` is ``None``.

    Ensures the nextaire_tools style is applied before any axes are created.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None`` a new figure and axes are
        created using ``figsize``.
    figsize : tuple of float, optional
        Figure size in inches, used only when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    _ensure_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)
    return fig, ax


_Ax = TypeVar("_Ax", Axes, np.ndarray)


def _finalize(
    fig: Figure,
    ax: _Ax,
    save_path: PathType | None = None,
    *,
    despine: bool = True,
    grid: bool = True,
) -> _Ax:
    """Apply spine/grid styling to one or more axes and optionally save.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure the axes belong to.
    ax : matplotlib.axes.Axes or numpy.ndarray of Axes
        A single axes or an array of axes (for small-multiple grids).
    save_path : str or os.PathLike, optional
        When given, the figure is saved with ``bbox_inches="tight"`` and
        ``dpi=150``.
    despine : bool, default True
        Hide the top/right spines and recolor the remaining spines.
    grid : bool, default True
        When ``True`` keep the (recessive) grid behind the data; when
        ``False`` turn the grid off (e.g. for heatmaps / image maps).

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray
        The same ``ax`` object passed in.
    """
    axes = np.ravel(ax) if isinstance(ax, np.ndarray) else [ax]
    for a in axes:
        if despine:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(INK["baseline"])
        if grid:
            a.set_axisbelow(True)
        else:
            a.grid(False)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return ax


@contextmanager
def nextaire_tools_style(context: str = "notebook", *, grid: bool = True) -> Iterator[None]:
    """Temporarily apply the nextaire_tools style within a ``with`` block.

    On exit the previous :data:`matplotlib.rcParams` are restored, so this is
    a non-invasive alternative to the global :func:`set_style`.

    Parameters
    ----------
    context : str, default "notebook"
        Seaborn plotting context.
    grid : bool, default True
        Whether to draw the background grid.

    Yields
    ------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> with nextaire_tools_style("talk"):
    ...     fig, ax = plt.subplots()
    ...     _ = ax.plot([0, 1], [0, 1])
    """
    with mpl.rc_context():
        set_style(context, grid=grid)
        yield
