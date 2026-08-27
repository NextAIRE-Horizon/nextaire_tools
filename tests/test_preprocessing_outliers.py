"""Tests for OutlierHandler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.preprocessing import OutlierHandler


@pytest.mark.parametrize("method", ["iqr", "zscore", "modified_zscore", "quantile"])
def test_bounds_learned_per_column(aq_df, method):
    step = OutlierHandler(columns=["no2", "pm10"], method=method).fit(aq_df)
    assert set(step.bounds_) == {"no2", "pm10"}
    for low, high in step.bounds_.values():
        assert low < high


def test_clip_keeps_shape_and_bounds(aq_df):
    step = OutlierHandler(columns=["pm10"], method="iqr", strategy="clip")
    out = step.fit_transform(aq_df)
    assert len(out) == len(aq_df)
    low, high = step.bounds_["pm10"]
    assert out["pm10"].max() <= high + 1e-6
    assert out["pm10"].min() >= low - 1e-6


def test_drop_removes_outlier_rows(aq_df):
    step = OutlierHandler(columns=["pm10", "no2"], method="iqr", strategy="drop")
    out = step.fit_transform(aq_df)
    assert len(out) < len(aq_df)


def test_nan_strategy_sets_nan(aq_df):
    step = OutlierHandler(columns=["pm10"], method="iqr", strategy="nan")
    out = step.fit_transform(aq_df)
    assert out["pm10"].isna().sum() >= 1
    assert len(out) == len(aq_df)


def test_flag_adds_column(aq_df):
    step = OutlierHandler(columns=["pm10"], method="iqr", strategy="flag")
    out = step.fit_transform(aq_df)
    assert "is_outlier" in out.columns
    assert out["is_outlier"].sum() >= 1


def test_isolation_forest_drop(aq_df):
    step = OutlierHandler(
        columns=["no2", "o3", "pm10"],
        method="isolation_forest",
        strategy="drop",
        contamination=0.05,
        random_state=0,
    )
    out = step.fit_transform(aq_df)
    assert len(out) <= len(aq_df)


def test_isolation_forest_clip_is_invalid(aq_df):
    with pytest.raises(ConfigurationError):
        OutlierHandler(method="isolation_forest", strategy="clip").fit(aq_df)


def test_outlier_fraction_reported(aq_df):
    step = OutlierHandler(columns=["pm10"], method="iqr").fit(aq_df)
    assert 0.0 <= step.outlier_fraction_ <= 1.0
    assert step.n_outliers_ >= 1


def test_invalid_method():
    with pytest.raises(ConfigurationError):
        OutlierHandler(method="nope").fit(pd.DataFrame({"a": np.arange(10.0)}))
