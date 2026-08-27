"""Shared pytest fixtures and a headless Matplotlib backend for the test suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display; must run before pyplot is imported anywhere

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def aq_df(rng: np.random.Generator) -> pd.DataFrame:
    """A small, hourly, datetime-indexed air-quality frame with injected NaNs and outliers."""
    idx = pd.date_range("2024-01-01", periods=720, freq="h", name="timestamp")
    n = len(idx)
    t = np.arange(n)
    no2 = 20 + 10 * np.sin(t * 2 * np.pi / 24) + rng.normal(0, 3, n)
    o3 = 40 + 15 * np.cos(t * 2 * np.pi / 24) + rng.normal(0, 4, n)
    pm10 = 15 + rng.gamma(2.0, 3.0, n)
    df = pd.DataFrame({"no2": no2, "o3": o3, "pm10": pm10}, index=idx)

    # Missing values
    df.iloc[5:8, df.columns.get_loc("no2")] = np.nan
    df.iloc[100, df.columns.get_loc("o3")] = np.nan

    # Clear outliers
    df.iloc[50, df.columns.get_loc("pm10")] = 500.0
    df.iloc[300, df.columns.get_loc("no2")] = -200.0
    return df


@pytest.fixture
def aq_csv(tmp_path, aq_df: pd.DataFrame):
    path = tmp_path / "aq.csv"
    aq_df.to_csv(path)
    return path


@pytest.fixture
def stations_df() -> pd.DataFrame:
    """A minimal stations frame in the shape produced by ``load_stations``."""
    return pd.DataFrame(
        {
            "station_name": ["Alpha", "Beta"],
            "station_lon": [15.98, 16.44],
            "station_lat": [45.81, 46.31],
        }
    )
