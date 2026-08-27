"""Tests for regression metrics and cross-validation reporting."""

from __future__ import annotations

import numpy as np
import pytest

from nextaire_tools.exceptions import SchemaError
from nextaire_tools.models import (
    BlockingTimeSeriesSplit,
    cross_val_report,
    make_regressor,
    regression_metrics,
)


def test_perfect_prediction_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["index_of_agreement"] == pytest.approx(1.0)
    assert m["fac2"] == pytest.approx(1.0)


def test_metrics_keys_present():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 100)
    yhat = y + rng.normal(0, 1, 100)
    m = regression_metrics(y, yhat)
    for key in [
        "mae",
        "mse",
        "rmse",
        "r2",
        "mape",
        "smape",
        "bias",
        "pearson_r",
        "spearman_r",
        "index_of_agreement",
        "fac2",
    ]:
        assert key in m


def test_metrics_handle_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    yhat = np.array([1.0, np.nan, 3.0, 4.0])
    m = regression_metrics(y, yhat)  # only indices 0 and 3 are usable
    assert m["mae"] == pytest.approx(0.0)


def test_metrics_length_mismatch_raises():
    with pytest.raises(SchemaError):
        regression_metrics([1, 2, 3], [1, 2])


def test_metrics_subset():
    m = regression_metrics([1.0, 2.0], [1.0, 2.0], metrics=["mae", "rmse"])
    assert set(m) == {"mae", "rmse"}


def test_cross_val_report_shape():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, n)
    model = make_regressor("linear")
    report = cross_val_report(model, X, y, cv=BlockingTimeSeriesSplit(n_splits=4))
    assert "mae" in report.columns
    assert "mean" in report.index
    assert "std" in report.index
