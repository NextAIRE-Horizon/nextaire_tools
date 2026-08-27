"""Visualization: a colorblind-safe, light/dark-aware Matplotlib theme plus EDA,
outlier-inspection, and model-evaluation figures.

Every plotting function accepts ``ax=None`` / ``figsize`` / ``save_path`` and
returns the Matplotlib ``Axes`` (or an array of them); none call ``plt.show()``.
"""

from __future__ import annotations

from nextaire_tools.viz.eda import (
    plot_correlation,
    plot_distributions,
    plot_missingness,
    plot_seasonality,
    plot_timeseries,
)
from nextaire_tools.viz.evaluation import (
    plot_feature_importance,
    plot_predictions,
    plot_residuals,
)
from nextaire_tools.viz.outliers import plot_boxplots, plot_outliers
from nextaire_tools.viz.style import (
    DIVERGING_CMAP,
    INK,
    PALETTE,
    SEQUENTIAL_CMAP,
    set_style,
)

__all__ = [
    "set_style",
    "PALETTE",
    "INK",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "plot_missingness",
    "plot_distributions",
    "plot_correlation",
    "plot_timeseries",
    "plot_seasonality",
    "plot_boxplots",
    "plot_outliers",
    "plot_predictions",
    "plot_residuals",
    "plot_feature_importance",
]
