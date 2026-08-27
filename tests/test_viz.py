"""Smoke tests for the visualization module (headless Agg backend via conftest)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from nextaire_tools.viz import (
    PALETTE,
    plot_boxplots,
    plot_correlation,
    plot_distributions,
    plot_feature_importance,
    plot_missingness,
    plot_outliers,
    plot_predictions,
    plot_residuals,
    plot_seasonality,
    plot_timeseries,
    set_style,
)


def _is_axes(obj) -> bool:
    if isinstance(obj, Axes):
        return True
    arr = np.asarray(obj).ravel()
    return len(arr) > 0 and all(isinstance(a, Axes) for a in arr)


def test_palette_has_eight_colors():
    assert len(PALETTE) == 8
    assert all(c.startswith("#") for c in PALETTE)


def test_set_style_is_idempotent():
    set_style()
    set_style(context="paper")


def test_eda_plots_return_axes(aq_df):
    assert _is_axes(plot_missingness(aq_df))
    assert _is_axes(plot_distributions(aq_df))
    assert _is_axes(plot_correlation(aq_df))
    assert _is_axes(plot_timeseries(aq_df))
    assert _is_axes(plot_seasonality(aq_df, column="o3", by="hour"))


def test_outlier_plots(aq_df):
    assert _is_axes(plot_boxplots(aq_df))
    assert _is_axes(plot_outliers(aq_df, column="pm10", bounds=(0.0, 100.0)))


def test_evaluation_plots():
    rng = np.random.default_rng(0)
    y = rng.normal(20, 5, 100)
    yhat = y + rng.normal(0, 2, 100)
    assert _is_axes(plot_predictions(y, yhat))
    assert _is_axes(plot_residuals(y, yhat))
    assert _is_axes(plot_feature_importance(np.array([0.5, 0.3, 0.2]), ["a", "b", "c"]))


def test_saves_file(aq_df, tmp_path):
    out = tmp_path / "fig.png"
    plot_timeseries(aq_df, save_path=out)
    assert out.exists()
