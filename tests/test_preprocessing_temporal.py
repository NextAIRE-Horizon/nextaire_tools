"""Tests for TemporalFeatures — the flagship cyclical/calendar feature step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import SchemaError
from nextaire_tools.preprocessing import TemporalFeatures


def test_adds_calendar_and_cyclical_columns(aq_df):
    step = TemporalFeatures(
        add=("hour", "dayofweek", "month", "is_weekend"),
        cyclical=("hour", "dayofweek", "dayofyear"),
    )
    out = step.fit_transform(aq_df)
    for col in ["hour", "dayofweek", "month", "is_weekend"]:
        assert col in out.columns
    for base in ["hour", "dayofweek", "dayofyear"]:
        assert f"{base}_sin" in out.columns
        assert f"{base}_cos" in out.columns


def test_cyclical_values_are_bounded(aq_df):
    step = TemporalFeatures(add=(), cyclical=("hour", "dayofyear"))
    out = step.fit_transform(aq_df)
    for col in ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]:
        assert out[col].abs().max() <= 1.0 + 1e-9


def test_hour_sin_cos_are_consistent(aq_df):
    step = TemporalFeatures(add=("hour",), cyclical=("hour",))
    out = step.fit_transform(aq_df)
    # sin^2 + cos^2 == 1
    unit = out["hour_sin"] ** 2 + out["hour_cos"] ** 2
    assert np.allclose(unit.to_numpy(), 1.0, atol=1e-9)


def test_is_weekend_matches_index(aq_df):
    step = TemporalFeatures(add=("is_weekend",), cyclical=())
    out = step.fit_transform(aq_df)
    expected = (aq_df.index.dayofweek >= 5).astype(int)
    assert np.array_equal(out["is_weekend"].to_numpy(), expected)


def test_works_from_time_column():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=48, freq="h"),
            "value": np.arange(48.0),
        }
    )
    step = TemporalFeatures(time_col="timestamp", add=("hour",), cyclical=("hour",))
    out = step.fit_transform(df)
    assert "hour_sin" in out.columns
    assert "timestamp" in out.columns  # original preserved


def test_missing_datetime_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(SchemaError):
        TemporalFeatures().fit_transform(df)


def test_drop_raw_cyclical(aq_df):
    step = TemporalFeatures(add=("hour",), cyclical=("hour",), drop_raw_cyclical=True)
    out = step.fit_transform(aq_df)
    assert "hour" not in out.columns
    assert "hour_sin" in out.columns


def test_feature_names_out(aq_df):
    step = TemporalFeatures(add=("hour",), cyclical=("hour",)).fit(aq_df)
    names = list(step.get_feature_names_out())
    assert "hour_sin" in names and "hour_cos" in names
