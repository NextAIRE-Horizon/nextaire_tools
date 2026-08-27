"""Tests for MissingValueHandler."""

from __future__ import annotations

import numpy as np
import pytest

from nextaire_tools.exceptions import ConfigurationError, NotFittedError
from nextaire_tools.preprocessing import MissingValueHandler


def test_drop_removes_rows_with_nan(aq_df):
    step = MissingValueHandler(strategy="drop")
    out = step.fit_transform(aq_df)
    assert out.isna().to_numpy().sum() == 0
    assert len(out) < len(aq_df)


def test_mean_imputation_fills_all(aq_df):
    step = MissingValueHandler(columns=["no2", "o3"], strategy="mean")
    out = step.fit_transform(aq_df)
    assert out[["no2", "o3"]].isna().to_numpy().sum() == 0
    assert len(out) == len(aq_df)  # imputation keeps rows


def test_interpolate_keeps_shape(aq_df):
    step = MissingValueHandler(strategy="interpolate", limit=3)
    out = step.fit_transform(aq_df)
    assert len(out) == len(aq_df)


def test_add_indicator_creates_flag_columns(aq_df):
    step = MissingValueHandler(columns=["no2"], strategy="mean", add_indicator=True)
    out = step.fit_transform(aq_df)
    assert any("no2" in str(c) and "missing" in str(c) for c in out.columns)


def test_column_missing_threshold_drops_column(aq_df):
    df = aq_df.copy()
    df["mostly_nan"] = np.nan
    df.iloc[0, df.columns.get_loc("mostly_nan")] = 1.0
    step = MissingValueHandler(strategy="mean", column_missing_threshold=0.5)
    out = step.fit_transform(df)
    assert "mostly_nan" not in out.columns
    assert "mostly_nan" in step.dropped_columns_


def test_invalid_strategy_raises():
    with pytest.raises(ConfigurationError):
        MissingValueHandler(strategy="bogus").fit_transform(
            __import__("pandas").DataFrame({"a": [1.0, None]})
        )


def test_transform_before_fit_raises(aq_df):
    with pytest.raises(NotFittedError):
        MissingValueHandler().transform(aq_df)


def test_does_not_mutate_input(aq_df):
    before = aq_df.copy()
    MissingValueHandler(strategy="mean").fit_transform(aq_df)
    # original untouched (same NaN pattern)
    assert aq_df.isna().to_numpy().sum() == before.isna().to_numpy().sum()
