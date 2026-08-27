"""Reproduction recipe for Petrić et al. (2024), AAQR 24:230317.

    Ensemble Machine Learning, Deep Learning, and Time Series Forecasting:
    Improving Prediction Accuracy for Hourly Concentrations of Ambient Air
    Pollutants. Aerosol and Air Quality Research 24, 230317.
    DOI: 10.4209/aaqr.230317

This script rebuilds the paper's *methodology* end-to-end on synthetic Graz-shaped
data (see ``_synthetic.py``). To reproduce the published numbers, replace
``make_graz_hourly`` with ``nextaire_tools.load_table(...)`` of the processed dataset
deposited on Zenodo (record 7959116, DOI 10.5281/zenodo.7959116).

Pipeline (Fig. 2 of the paper), each step mapped to a nextaire_tools building block:
  1. Winsorise short-lived spikes  -> OutlierHandler(method="rolling_sigma", window=72, sigma=4)
  2. Multivariate imputation       -> MissingValueHandler(strategy="iterative")
  3. Wind direction -> x/y, speed  -> WindDecomposer
  4. 12-hour median lag features   -> LagFeatures(windows=[12], agg="median")
  5. Drop >90% correlated features -> CorrelationFilter(threshold=0.9)
  6. One-year temporal hold-out    -> temporal_train_test_split
  7. Models: RF, (MLP/LSTM/CNN if torch), Prophet + Hybrid (if prophet)
  8. Metrics: R^2, nMAE, nRMSE (IQR-normalised)

Run:  python reproductions/paper1_petric2024_aaqr.py
"""

from __future__ import annotations

from _synthetic import make_graz_hourly

from nextaire_tools import (
    CorrelationFilter,
    LagFeatures,
    MissingValueHandler,
    OutlierHandler,
    Pipeline,
    Scaler,
    TemporalFeatures,
    WindDecomposer,
)
from nextaire_tools.models import make_regressor, regression_metrics, temporal_train_test_split

TARGET = "no2"
METRICS = ("r2", "nmae", "nrmse", "index_of_agreement", "fac2")


def build_pipeline() -> Pipeline:
    """The paper's preprocessing chain as a reproducible nextaire_tools Pipeline."""
    return Pipeline(
        [
            # Winsorise on a rolling 3-day (72 h) window at +/- 4 sigma.
            OutlierHandler(
                columns=["no", "no2", "o3", "pm10"],
                method="rolling_sigma",
                strategy="clip",
                window=72,
                sigma=4.0,
            ),
            # Round-robin multivariate imputation (iterative imputer).
            MissingValueHandler(strategy="iterative", max_iter=10, random_state=0),
            # Wind direction -> stable x/y; 10 m speed already provided.
            WindDecomposer(direction_col="wind_dir", drop_original=True),
            # Calendar + cyclical encodings (day-of-week, day-of-year, month).
            TemporalFeatures(
                add=("hour", "dayofweek", "month"),
                cyclical=("hour", "dayofweek", "dayofyear"),
            ),
            # "Median of the 12 most recent measurements" causal lag features.
            LagFeatures(
                columns=["no", "o3", "pm10", "temp", "wind_speed", "blh"],
                windows=[12],
                agg="median",
            ),
            # Backfill the few leading rows the causal lag window cannot cover.
            MissingValueHandler(strategy="bfill"),
            # Remove collinear features (>90% absolute correlation); keep the target.
            CorrelationFilter(threshold=0.9, protect=[TARGET]),
            Scaler(method="standard"),
        ]
    )


def main() -> None:
    raw = make_graz_hourly(n_days=365, seed=0)
    clean = build_pipeline().fit_transform(raw)

    train, test = temporal_train_test_split(clean, test_size=0.25)
    x_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    x_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    print(f"Reproduction: Petrić et al. (2024) — target = {TARGET}")
    print(f"  features={x_train.shape[1]}  train={len(train)}  test={len(test)}\n")

    results: dict[str, dict[str, float]] = {}

    # --- Random Forest (the paper's ensemble baseline) --------------------
    rf = make_regressor("random_forest", n_estimators=400)
    rf.fit(x_train, y_train)
    results["RF"] = regression_metrics(y_test, rf.predict(x_test), metrics=METRICS)

    # --- Deep models (optional torch) -------------------------------------
    try:
        from nextaire_tools.models import CNNRegressor, LSTMRegressor, MLPRegressor

        for name, model in {
            "MLP": MLPRegressor(hidden_sizes=(64, 32), epochs=15),
            "LSTM": LSTMRegressor(window=24, hidden_size=48, epochs=8),
            "CNN": CNNRegressor(window=24, channels=(32, 32), epochs=8),
        }.items():
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            pred = model.predict(x_test.to_numpy())
            # Windowed models return one prediction per window; align tails.
            y_tail = y_test.to_numpy()[-len(pred):]
            results[name] = regression_metrics(y_tail, pred, metrics=METRICS)
    except ImportError:
        print("  (torch not installed - skipping MLP/LSTM/CNN; pip install 'nextaire_tools[deep]')\n")

    # --- Prophet + Hybrid RF (optional prophet) ---------------------------
    try:
        from nextaire_tools.models import HybridProphetRegressor

        hyb = HybridProphetRegressor(
            base_estimator=make_regressor("random_forest", n_estimators=400)
        )
        hyb.fit(x_train, y_train)
        results["HYB"] = regression_metrics(y_test, hyb.predict(x_test), metrics=METRICS)
    except ImportError:
        print("  (prophet not installed - skipping Prophet/Hybrid; pip install nextaire_tools[forecast])\n")

    # --- Report -----------------------------------------------------------
    header = f"{'model':<8}" + "".join(f"{m:>20}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<8}" + "".join(f"{scores[m]:>20.4f}" for m in METRICS)
        print(row)


if __name__ == "__main__":
    main()
