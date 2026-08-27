"""Reproduction recipe for Jiménez-Navarro et al. (2024), Results in Engineering 24:103290.

    Explainable deep learning on multi-target time series forecasting: An air
    pollution use case. Results in Engineering 24, 103290.
    DOI: 10.1016/j.rineng.2024.103290

The paper's headline contribution is the Temporal Selection Layer (TSL), an
embedded feature-selection layer the authors add to feed-forward / LSTM nets —
that specific layer is the paper's own research artifact and is *not* bundled in
nextaire_tools. What nextaire_tools reproduces here is everything around it: the multi-station
Graz dataset, Bayesian-Ridge iterative imputation, 12-hour median lag features,
wind decomposition, the *blocked* time-series cross-validation, the baseline
model zoo (Decision Tree, Lasso, KNN, Random Forest, XGBoost, LSTM), and the
paper's forecasting metrics (MAE, RMSE, WAPE).

Data: Austrian Styria open portal + ERA5-Land (CDS API). Replace
``make_graz_hourly`` with ``nextaire_tools.load_table(...)`` of the real data to
reproduce the published numbers.

Run:  python reproductions/paper2_jimenez2024_multitarget.py
"""

from __future__ import annotations

import numpy as np
from _synthetic import make_graz_hourly
from sklearn.linear_model import BayesianRidge

from nextaire_tools import (
    LagFeatures,
    MissingValueHandler,
    Pipeline,
    Scaler,
    TemporalFeatures,
    WindDecomposer,
)
from nextaire_tools.models import BlockingTimeSeriesSplit, make_regressor, regression_metrics

# Multi-target: the paper forecasts several station-pollutant series jointly.
TARGETS = ("no2", "o3", "pm10")
METRICS = ("mae", "rmse", "wape")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # Iterative imputation with a Bayesian-Ridge estimator (the paper's choice).
            MissingValueHandler(strategy="iterative", estimator=BayesianRidge(), random_state=0),
            WindDecomposer(direction_col="wind_dir", drop_original=True),
            TemporalFeatures(add=("hour", "dayofweek", "month"), cyclical=("hour", "dayofweek")),
            LagFeatures(columns=["no", "temp", "wind_speed", "blh"], windows=[12], agg="median"),
            MissingValueHandler(strategy="bfill"),
            Scaler(method="standard"),
        ]
    )


def blocked_cv_score(model_name: str, X, y, n_splits: int = 4) -> dict[str, float]:
    """Average metrics over a blocked (non-overlapping) time-series CV."""
    cv = BlockingTimeSeriesSplit(n_splits=n_splits)
    per_fold: list[dict[str, float]] = []
    for train_idx, test_idx in cv.split(X):
        model = make_regressor(model_name, **_model_kwargs(model_name))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        per_fold.append(regression_metrics(y.iloc[test_idx], pred, metrics=METRICS))
    return {m: float(np.mean([f[m] for f in per_fold])) for m in METRICS}


def _model_kwargs(name: str) -> dict:
    return {
        "decision_tree": {"max_depth": 8},
        "lasso": {"alpha": 0.01},
        "knn": {"n_neighbors": 15},
        "random_forest": {"n_estimators": 200},
        "xgboost": {"n_estimators": 100, "max_depth": 4},
    }.get(name, {})


def main() -> None:
    clean = build_pipeline().fit_transform(make_graz_hourly(n_days=365, seed=1))

    candidates = ["decision_tree", "lasso", "knn", "random_forest"]
    try:  # XGBoost is optional (pip install 'nextaire_tools[boost]')
        make_regressor("xgboost")
        candidates.append("xgboost")
    except Exception:
        print("  (xgboost not installed — skipping; pip install 'nextaire_tools[boost]')\n")

    print("Reproduction: Jiménez-Navarro et al. (2024) — blocked CV, multi-target")
    n_feat = clean.shape[1] - len(TARGETS)
    print(f"  targets={list(TARGETS)}  features~{n_feat}  rows={len(clean)}\n")

    header = f"{'model':<16}" + "".join(f"{m.upper():>10}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name in candidates:
        # Average each metric across the multiple target series (multi-target).
        scores = {m: [] for m in METRICS}
        for target in TARGETS:
            X = clean.drop(columns=list(TARGETS))
            s = blocked_cv_score(name, X, clean[target])
            for m in METRICS:
                scores[m].append(s[m])
        row = f"{name:<16}" + "".join(f"{np.mean(scores[m]):>10.4f}" for m in METRICS)
        print(row)

    print("\nNote: the paper's Temporal Selection Layer (TSL) is its own research")
    print("contribution and is not part of nextaire_tools; the LSTM/FF baselines are available")
    print("via nextaire_tools.models.LSTMRegressor / MLPRegressor with the '[deep]' extra.")


if __name__ == "__main__":
    main()
