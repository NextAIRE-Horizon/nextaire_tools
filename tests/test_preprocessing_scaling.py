"""Tests for the Scaler step."""

from __future__ import annotations

import numpy as np
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.preprocessing import Scaler


def test_standard_scaling_zero_mean_unit_std(aq_df):
    df = aq_df.dropna()
    step = Scaler(columns=["no2", "o3"], method="standard")
    out = step.fit_transform(df)
    assert np.allclose(out["no2"].mean(), 0.0, atol=1e-6)
    assert np.allclose(out["no2"].std(ddof=0), 1.0, atol=1e-6)


def test_only_selected_columns_changed(aq_df):
    df = aq_df.dropna()
    step = Scaler(columns=["no2"], method="minmax")
    out = step.fit_transform(df)
    # pm10 unchanged
    assert np.allclose(out["pm10"].to_numpy(), df["pm10"].to_numpy())
    # no2 within [0, 1]
    assert out["no2"].min() >= -1e-9
    assert out["no2"].max() <= 1 + 1e-9


def test_inverse_transform_roundtrip(aq_df):
    df = aq_df.dropna()
    step = Scaler(columns=["no2", "o3"], method="standard")
    scaled = step.fit_transform(df)
    restored = step.inverse_transform(scaled)
    assert np.allclose(restored["no2"].to_numpy(), df["no2"].to_numpy(), atol=1e-6)


def test_invalid_method_raises(aq_df):
    with pytest.raises(ConfigurationError):
        Scaler(method="unknown").fit(aq_df.dropna())
