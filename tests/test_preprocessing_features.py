"""Tests for the derived-feature steps: WindDecomposer, LagFeatures, CorrelationFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import ConfigurationError, NotFittedError, SchemaError
from nextaire_tools.preprocessing.features import (
    CorrelationFilter,
    LagFeatures,
    WindDecomposer,
)


@pytest.fixture
def wind_df() -> pd.DataFrame:
    """A datetime-indexed frame with a wind-direction column and (u, v) vector."""
    idx = pd.date_range("2024-01-01", periods=6, freq="h", name="timestamp")
    return pd.DataFrame(
        {
            "wd": [0.0, 90.0, 180.0, 270.0, 360.0, 45.0],
            "u": [1.0, 0.0, -1.0, 0.0, 3.0, -2.0],
            "v": [0.0, 1.0, 0.0, -1.0, 4.0, 2.0],
            "no2": [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
        },
        index=idx,
    )


# --------------------------------------------------------------- WindDecomposer
def test_wind_direction_decomposition(wind_df):
    step = WindDecomposer(direction_col="wd")
    out = step.fit_transform(wind_df)
    assert "wd_x" in out.columns and "wd_y" in out.columns
    rad = wind_df["wd"] * np.pi / 180.0
    assert np.allclose(out["wd_x"].to_numpy(), np.cos(rad))
    assert np.allclose(out["wd_y"].to_numpy(), np.sin(rad))
    # 0 deg and 360 deg map to the same point (no discontinuity).
    assert np.isclose(out["wd_x"].iloc[0], out["wd_x"].iloc[4])
    assert np.isclose(out["wd_y"].iloc[0], out["wd_y"].iloc[4])


def test_wind_original_retained_by_default(wind_df):
    out = WindDecomposer(direction_col="wd").fit_transform(wind_df)
    assert "wd" in out.columns


def test_wind_drop_original(wind_df):
    step = WindDecomposer(direction_col="wd", speed_from_uv=("u", "v"), drop_original=True)
    out = step.fit_transform(wind_df)
    for col in ["wd", "u", "v"]:
        assert col not in out.columns
    assert "wd_x" in out.columns and "wind_speed" in out.columns
    assert "no2" in out.columns  # unrelated column untouched


def test_wind_speed_from_uv(wind_df):
    out = WindDecomposer(speed_from_uv=("u", "v")).fit_transform(wind_df)
    expected = np.sqrt(wind_df["u"] ** 2 + wind_df["v"] ** 2)
    assert np.allclose(out["wind_speed"].to_numpy(), expected)


def test_wind_custom_speed_name(wind_df):
    out = WindDecomposer(speed_from_uv=("u", "v"), speed_out="ws").fit_transform(wind_df)
    assert "ws" in out.columns
    assert "wind_speed" not in out.columns


def test_wind_index_preserved(wind_df):
    out = WindDecomposer(direction_col="wd").fit_transform(wind_df)
    assert isinstance(out.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(out.index, wind_df.index)


def test_wind_requires_one_source(wind_df):
    with pytest.raises(ConfigurationError):
        WindDecomposer().fit_transform(wind_df)


def test_wind_missing_column_raises(wind_df):
    with pytest.raises(SchemaError):
        WindDecomposer(direction_col="does_not_exist").fit_transform(wind_df)
    with pytest.raises(SchemaError):
        WindDecomposer(speed_from_uv=("u", "nope")).fit_transform(wind_df)


def test_wind_feature_names_out(wind_df):
    step = WindDecomposer(direction_col="wd", speed_from_uv=("u", "v"), drop_original=True).fit(
        wind_df
    )
    names = list(step.get_feature_names_out())
    assert names == ["no2", "wd_x", "wd_y", "wind_speed"]


def test_wind_not_fitted_raises(wind_df):
    with pytest.raises(NotFittedError):
        WindDecomposer(direction_col="wd").transform(wind_df)


# ------------------------------------------------------------------ LagFeatures
def test_lag_columns_and_values():
    df = pd.DataFrame({"no2": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = LagFeatures(columns="no2", lags=(1, 2)).fit_transform(df)
    assert list(out.columns) == ["no2", "no2_lag1", "no2_lag2"]
    assert np.array_equal(
        out["no2_lag1"].to_numpy(), np.array([np.nan, 1.0, 2.0, 3.0, 4.0]), equal_nan=True
    )
    assert np.array_equal(
        out["no2_lag2"].to_numpy(),
        np.array([np.nan, np.nan, 1.0, 2.0, 3.0]),
        equal_nan=True,
    )


def test_lag_original_columns_retained(aq_df):
    out = LagFeatures(lags=(1,)).fit_transform(aq_df)
    for col in aq_df.columns:
        assert col in out.columns


def test_rolling_is_causal_no_leak():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    out = LagFeatures(columns="x", windows=(2,), agg="mean").fit_transform(df)
    roll = out["x_roll2_mean"].to_numpy()
    # Row 0 has no preceding data -> NaN. Row i uses only rows < i (causal).
    assert np.isnan(roll[0])
    assert roll[1] == pytest.approx(10.0)  # mean of {10}
    assert roll[2] == pytest.approx(15.0)  # mean of {10, 20}
    assert roll[3] == pytest.approx(25.0)  # mean of {20, 30}
    # The current row's value never appears in its own rolling feature.
    assert roll[3] != pytest.approx(40.0)


def test_rolling_median_matches_recent_window():
    df = pd.DataFrame({"x": [1.0, 100.0, 2.0, 3.0, 4.0]})
    out = LagFeatures(columns="x", windows=(3,), agg="median").fit_transform(df)
    roll = out["x_roll3_median"].to_numpy()
    assert np.isnan(roll[0])
    assert roll[4] == pytest.approx(np.median([100.0, 2.0, 3.0]))


def test_lag_default_all_numeric(aq_df):
    out = LagFeatures(lags=(1,)).fit_transform(aq_df)
    for col in ["no2", "o3", "pm10"]:
        assert f"{col}_lag1" in out.columns


def test_lag_requires_lags_or_windows():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ConfigurationError):
        LagFeatures().fit_transform(df)


def test_lag_invalid_agg():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ConfigurationError):
        LagFeatures(windows=(2,), agg="sum").fit_transform(df)


def test_lag_feature_names_out():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    step = LagFeatures(columns=["a"], lags=(1,), windows=(2,), agg="median").fit(df)
    names = list(step.get_feature_names_out())
    assert names == ["a", "b", "a_lag1", "a_roll2_median"]


def test_lag_index_preserved(aq_df):
    out = LagFeatures(lags=(1,)).fit_transform(aq_df)
    assert isinstance(out.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(out.index, aq_df.index)


def test_lag_not_fitted_raises():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(NotFittedError):
        LagFeatures(lags=(1,)).transform(df)


# ------------------------------------------------------------ CorrelationFilter
def test_correlation_filter_drops_collinear():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated with a
            "c": [5.0, 1.0, 4.0, 2.0, 3.0],  # uncorrelated
        }
    )
    step = CorrelationFilter(threshold=0.9).fit(df)
    out = step.transform(df)
    assert "a" in out.columns
    assert "b" not in out.columns  # dropped as collinear with a
    assert "c" in out.columns
    assert step.dropped_columns_ == ["b"]
    assert "a" in step.kept_columns_ and "c" in step.kept_columns_


def test_correlation_filter_honors_protect():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    # Protect b: since a is first-kept and b is protected, nothing is dropped.
    step = CorrelationFilter(threshold=0.9, protect="b").fit(df)
    out = step.transform(df)
    assert "b" in out.columns
    assert "b" not in step.dropped_columns_


def test_correlation_filter_non_candidate_always_kept():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    # Only 'b' is a candidate; 'a' is outside the candidate set and always kept.
    step = CorrelationFilter(threshold=0.9, columns=["b"]).fit(df)
    out = step.transform(df)
    assert "a" in out.columns
    assert "b" in out.columns  # nothing to compare against -> kept
    assert step.dropped_columns_ == []


def test_correlation_filter_threshold_validation():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ConfigurationError):
        CorrelationFilter(threshold=0.0).fit(df)
    with pytest.raises(ConfigurationError):
        CorrelationFilter(threshold=1.5).fit(df)


def test_correlation_filter_feature_names_out():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
    )
    step = CorrelationFilter(threshold=0.9).fit(df)
    names = list(step.get_feature_names_out())
    assert names == ["a", "c"]


def test_correlation_filter_index_preserved(aq_df):
    out = CorrelationFilter(threshold=0.99).fit_transform(aq_df)
    assert isinstance(out.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(out.index, aq_df.index)


def test_correlation_filter_not_fitted_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(NotFittedError):
        CorrelationFilter().transform(df)
