"""Tests for model-interpretation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from nextaire_tools.models.interpret import (
    permutation_importance_report,
    shap_importance,
    tree_shap_values,
)


def _obvious_signal(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """y depends only on the 'strong' column; 'noise' is irrelevant."""
    rng = np.random.default_rng(seed)
    n = 300
    strong = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"strong": strong, "noise": noise})
    y = 3.0 * strong + rng.normal(0, 0.01, n)
    return X, y


def test_permutation_importance_report_dataframe():
    X, y = _obvious_signal()
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
    report = permutation_importance_report(model, X, y, n_repeats=5, random_state=0)

    assert list(report.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(report["feature"]) == {"strong", "noise"}
    # Sorted descending, and the informative feature ranks first.
    assert report.iloc[0]["feature"] == "strong"
    assert report["importance_mean"].is_monotonic_decreasing
    assert report.iloc[0]["importance_mean"] > report.iloc[1]["importance_mean"]


def test_permutation_importance_report_numpy_feature_names():
    X, y = _obvious_signal()
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X.to_numpy(), y)
    report = permutation_importance_report(model, X.to_numpy(), y, n_repeats=3)
    assert set(report["feature"]) == {"feature_0", "feature_1"}


def test_shap_importance():
    pytest.importorskip("shap")
    X, y = _obvious_signal()
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)

    report = shap_importance(model, X)
    assert list(report.columns) == ["feature", "mean_abs_shap"]
    assert set(report["feature"]) == {"strong", "noise"}
    assert report.iloc[0]["feature"] == "strong"
    assert report["mean_abs_shap"].is_monotonic_decreasing


def test_tree_shap_values_shape():
    pytest.importorskip("shap")
    X, y = _obvious_signal()
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    values = np.asarray(tree_shap_values(model, X))
    assert values.shape[0] == len(X)
    assert values.shape[1] == X.shape[1]
