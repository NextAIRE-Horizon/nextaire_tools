"""Model-evaluation charts for regression on air-quality targets.

Predicted-vs-observed scatter (or overlaid time series), residual diagnostics,
and feature-importance bars. Inputs are array-like (NumPy arrays,
:class:`pandas.Series`, or plain sequences); ``plot_feature_importance`` also
accepts a fitted estimator and reads its ``feature_importances_`` or ``coef_``.
No function mutates its inputs or calls ``plt.show``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score

from nextaire_tools._typing import PathType
from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.viz.style import INK, PALETTE, _ensure_style, _finalize, _prepare_ax

__all__ = [
    "plot_feature_importance",
    "plot_predictions",
    "plot_residuals",
]

_LOG = get_logger(__name__)


def _to_1d(values: ArrayLike, name: str) -> np.ndarray:
    """Coerce an array-like to a 1-D float :class:`numpy.ndarray`."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ConfigurationError(f"{name} is empty.")
    return arr


def _check_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ConfigurationError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )


def plot_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    index: ArrayLike | None = None,
    kind: str = "scatter",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Compare predictions against observations.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values (same length).
    index : array-like, optional
        X-axis values for ``kind="timeseries"``. When ``None`` the index of a
        :class:`pandas.Series` ``y_true`` is used, else a positional range.
    kind : {"scatter", "timeseries"}, default "scatter"
        ``"scatter"`` plots predicted vs observed with a dashed 1:1 line and
        the R2 score in the title. ``"timeseries"`` overlays both series.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to the underlying scatter/plot call.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ConfigurationError
        If ``kind`` is invalid or the inputs differ in length.

    Examples
    --------
    >>> ax = plot_predictions(y_test, preds, kind="scatter")  # doctest: +SKIP
    """
    if kind not in {"scatter", "timeseries"}:
        raise ConfigurationError(f"kind must be 'scatter' or 'timeseries', got {kind!r}.")

    yt = _to_1d(y_true, "y_true")
    yp = _to_1d(y_pred, "y_pred")
    _check_lengths(yt, yp)

    fig, ax = _prepare_ax(ax, figsize)

    if kind == "scatter":
        ax.scatter(yt, yp, s=18, color=PALETTE[0], alpha=0.7, edgecolor="none", **kwargs)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=INK["baseline"], linewidth=1.5)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predictions (R² = {r2_score(yt, yp):.3f})")
        return _finalize(fig, ax, save_path)

    # kind == "timeseries"
    if index is not None:
        x: ArrayLike = np.asarray(index)
    elif isinstance(y_true, pd.Series):
        x = y_true.index.to_numpy()
    else:
        x = np.arange(yt.size)
    ax.plot(x, yt, color=PALETTE[0], label="Observed", **kwargs)
    ax.plot(x, yp, color=PALETTE[5], label="Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    return _finalize(fig, ax, save_path)


def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Draw residual diagnostics as two panels.

    Panel 1 is residuals versus predicted values with a dashed zero line;
    panel 2 is a histogram of the residuals.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values (same length).
    ax : numpy.ndarray of matplotlib.axes.Axes, optional
        A length-2 array of axes to draw on. When ``None`` a 1x2 grid is
        created.
    figsize : tuple of float, optional
        Figure size; defaults to ``(10, 4)``.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :func:`seaborn.histplot` for the distribution panel.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two-panel axes array.

    Raises
    ------
    ConfigurationError
        If the inputs differ in length or fewer than two axes are supplied.

    Examples
    --------
    >>> axes = plot_residuals(y_test, preds)  # doctest: +SKIP
    """
    yt = _to_1d(y_true, "y_true")
    yp = _to_1d(y_pred, "y_pred")
    _check_lengths(yt, yp)
    residuals = yt - yp

    _ensure_style()
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (10.0, 4.0))
    else:
        axes = ax if isinstance(ax, np.ndarray) else np.array([ax], dtype=object)
        if axes.size < 2:
            raise ConfigurationError("plot_residuals requires two axes.")
        fig = axes.ravel()[0].figure
    flat = axes.ravel()
    a_scatter, a_hist = flat[0], flat[1]

    a_scatter.scatter(yp, residuals, s=18, color=PALETTE[0], alpha=0.7, edgecolor="none")
    a_scatter.axhline(0.0, linestyle="--", color=INK["baseline"], linewidth=1.5)
    a_scatter.set_xlabel("Predicted")
    a_scatter.set_ylabel("Residual")
    a_scatter.set_title("Residuals vs. predicted")

    sns.histplot(residuals, kde=True, ax=a_hist, color=PALETTE[0], **kwargs)
    a_hist.set_xlabel("Residual")
    a_hist.set_title("Residual distribution")

    fig.tight_layout()
    _finalize(fig, axes, save_path)
    return axes


def _extract_importances(
    importances: Any,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Extract an importance vector and feature names from an estimator/array.

    Accepts a fitted estimator (reads ``feature_importances_`` or
    ``abs(coef_)``) or a raw array-like. When ``coef_`` is 2-D (multi-output)
    the mean absolute coefficient across outputs is used.
    """
    if hasattr(importances, "feature_importances_"):
        imp = np.asarray(importances.feature_importances_, dtype=float).ravel()
    elif hasattr(importances, "coef_"):
        coef = np.abs(np.asarray(importances.coef_, dtype=float))
        imp = coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
    else:
        imp = np.asarray(importances, dtype=float).ravel()

    if feature_names is not None:
        names = [str(n) for n in feature_names]
    elif hasattr(importances, "feature_names_in_"):
        names = [str(n) for n in importances.feature_names_in_]
    else:
        names = [f"feature_{i}" for i in range(imp.size)]

    if len(names) != imp.size:
        raise ConfigurationError(f"Got {imp.size} importances but {len(names)} feature names.")
    return imp, names


def plot_feature_importance(
    importances: Any,
    feature_names: Sequence[str] | None = None,
    *,
    top_n: int = 20,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: PathType | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a horizontal bar chart of the largest feature importances.

    Parameters
    ----------
    importances : estimator or array-like
        A fitted estimator exposing ``feature_importances_`` or ``coef_``, or
        a raw array of importance values.
    feature_names : sequence of str, optional
        Names aligned with the importances. When ``None`` they are taken from
        the estimator's ``feature_names_in_`` if available, else generated.
    top_n : int, default 20
        Show only the ``top_n`` largest importances.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if ``None``.
    figsize : tuple of float, optional
        Figure size.
    save_path : str or os.PathLike, optional
        If given, save the figure and return.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.barh`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ConfigurationError
        If the number of names does not match the number of importances.

    Examples
    --------
    >>> ax = plot_feature_importance(rf, top_n=15)  # doctest: +SKIP
    """
    imp, names = _extract_importances(importances, feature_names)

    top_n = max(1, int(top_n))
    order = np.argsort(imp)[::-1][:top_n]
    imp_top = imp[order]
    names_top = [names[i] for i in order]

    fig, ax = _prepare_ax(ax, figsize)
    # Largest at the top: assign descending y-positions to the sorted values.
    positions = np.arange(len(imp_top))[::-1]
    ax.barh(positions, imp_top, color=PALETTE[0], **kwargs)
    ax.set_yticks(positions)
    ax.set_yticklabels(names_top)
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    return _finalize(fig, ax, save_path)
