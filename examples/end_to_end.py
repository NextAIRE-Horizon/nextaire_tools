"""End-to-end nextaire_tools example on synthetic data (no network required).

Run with::

    python examples/end_to_end.py

It generates a synthetic hourly air-quality dataset, runs a cleaning + feature
pipeline, evaluates a model with a leakage-free time-series split, and writes a
few figures to ``examples/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures without a display

import numpy as np
import pandas as pd

from nextaire_tools import Pipeline, load_table, save_table
from nextaire_tools.models import (
    BlockingTimeSeriesSplit,
    cross_val_report,
    make_regressor,
)
from nextaire_tools.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    Scaler,
    TemporalFeatures,
)
from nextaire_tools.viz import (
    plot_correlation,
    plot_predictions,
    plot_seasonality,
    plot_timeseries,
)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"


def make_synthetic_station(n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Create a realistic hourly station frame with diurnal cycles, gaps, and spikes."""
    idx = pd.date_range("2023-01-01", periods=24 * n_days, freq="h", name="timestamp")
    n = len(idx)
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    diurnal = np.sin(t * 2 * np.pi / 24)
    weekly = np.sin(t * 2 * np.pi / (24 * 7))
    df = pd.DataFrame(
        {
            "no2": 28 + 12 * diurnal + 6 * weekly + rng.normal(0, 4, n),
            "o3": 45 - 10 * diurnal + rng.normal(0, 5, n),
            "pm10": 20 + 8 * weekly + rng.gamma(2.0, 3.0, n),
        },
        index=idx,
    )
    # Inject missing values and a couple of sensor spikes.
    df.iloc[50:53, df.columns.get_loc("no2")] = np.nan
    df.iloc[1000, df.columns.get_loc("pm10")] = 850.0
    return df


def main() -> None:
    FIGS.mkdir(exist_ok=True)

    # 1. Create + persist a raw CSV, then load it the way a user would.
    raw = make_synthetic_station()
    csv_path = HERE / "sample_station.csv"
    save_table(raw, csv_path)
    df = load_table(csv_path, time_col="timestamp", set_time_index=True)
    print(f"Loaded {len(df)} hourly rows, columns={list(df.columns)}")

    # 2. Exploratory figures.
    plot_timeseries(df, columns=["no2", "o3"], save_path=FIGS / "timeseries.png")
    plot_correlation(df, save_path=FIGS / "correlation.png")
    plot_seasonality(df, column="o3", by="hour", save_path=FIGS / "o3_by_hour.png")

    # 3. Reproducible cleaning + feature-engineering pipeline.
    pipe = Pipeline(
        [
            MissingValueHandler(strategy="interpolate"),
            OutlierHandler(columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"),
            TemporalFeatures(
                add=("hour", "dayofweek", "month", "is_weekend"),
                cyclical=("hour", "dayofweek", "dayofyear"),
            ),
            Scaler(columns=["no2", "o3", "pm10"], method="standard"),
        ]
    )
    clean = pipe.fit_transform(df)
    print(f"Feature-engineered frame: {clean.shape[1]} columns; NaNs = {int(clean.isna().sum().sum())}")

    # 4. Model NO2 with a leakage-free blocking time-series CV.
    target = "no2"
    X = clean.drop(columns=[target])
    y = clean[target]
    model = make_regressor("random_forest", n_estimators=200)
    report = cross_val_report(model, X, y, cv=BlockingTimeSeriesSplit(n_splits=5))
    print("\nCross-validation report:")
    print(report.round(3))

    # 5. Fit on all data and plot predictions vs observed.
    model.fit(X, y)
    plot_predictions(y, model.predict(X), kind="scatter", save_path=FIGS / "predictions.png")

    print(f"\nFigures written to {FIGS}")


if __name__ == "__main__":
    main()
