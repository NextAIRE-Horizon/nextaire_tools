"""Reproduction recipe for Račić et al. (2026), Atmospheric Environment: X 29:100413.

    Assessment of contributors to airborne PAHs and heavy metals in PM10 using
    temporal, spatial, traffic and heating data in explainable machine learning
    models. Atmospheric Environment: X 29, 100413.
    DOI: 10.1016/j.aeaoa.2026.100413

Method (NMF + Random Forest + SHAP), each mapped to a nextaire_tools building block:
  1. Assemble daily PM10-bound PAH/metal data + meteo + traffic + heating + station
  2. NMF factor analysis (rank 2), per species group   -> NMFApportionment
  3. One Random Forest per pollutant on log-concentration, 80/20 split,
     5-fold grid search                                 -> make_regressor + GridSearchCV
  4. Report held-out R^2                                -> regression_metrics
  5. Explain with TreeSHAP                              -> shap_importance (optional 'shap')

Data is available on request from the authors; here we use synthetic
Zagreb-shaped data (see ``_synthetic.py``). Replace ``make_zagreb_daily`` with
``nextaire_tools.load_table(...)`` of the real data to reproduce the published numbers.

Run:  python reproductions/paper3_racic2026_source_apportionment.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _synthetic import METALS, PAHS, ZAGREB_STATIONS, make_zagreb_daily
from sklearn.model_selection import GridSearchCV, train_test_split

from nextaire_tools.models import NMFApportionment, make_regressor, regression_metrics

FEATURES = [
    "temp", "temp_min", "temp_max", "radiation", "rh", "wind_speed",
    "pm10", "no2", "gas_sum", "traffic", "julian", "dow", "month",
]
POLLUTANTS = list(PAHS) + list(METALS)


def add_station_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the monitoring station as binary spatial features (the paper's proxy)."""
    dummies = pd.get_dummies(df["station"], prefix="station").astype(float)
    return pd.concat([df, dummies], axis=1)


def rf_per_pollutant(df: pd.DataFrame, pollutant: str) -> float:
    """Train one RF on log-concentration; return held-out R^2 (paper's Table 2)."""
    station_cols = [c for c in df.columns if c.startswith("station_")]
    X = df[FEATURES + station_cols].to_numpy()
    y = np.log(df[pollutant].to_numpy())  # log-transform skewed concentrations

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    grid = GridSearchCV(
        make_regressor("random_forest", n_jobs=-1),
        param_grid={"n_estimators": [100, 200], "max_depth": [None, 12]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    return regression_metrics(y_te, grid.best_estimator_.predict(X_te), metrics=["r2"])["r2"]


def nmf_sources(df: pd.DataFrame, species: list[str], station: str) -> pd.DataFrame:
    """Rank-2 NMF loadings for one species group at one station (Figs. 3-4)."""
    sub = df.loc[df["station"] == station, species]
    model = NMFApportionment(n_components=2, scale=True, random_state=0, max_iter=1000)
    model.fit(sub)
    return model.loadings()


def main() -> None:
    df = add_station_dummies(make_zagreb_daily(seed=0))
    print("Reproduction: Račić et al. (2026) — NMF + Random Forest + SHAP")
    print(f"  stations={list(ZAGREB_STATIONS)}  pollutants={len(POLLUTANTS)}  rows={len(df)}\n")

    # --- NMF source patterns (rank 2), PAHs at the first station ----------
    loadings = nmf_sources(df, list(PAHS), ZAGREB_STATIONS[0])
    print(f"NMF rank-2 PAH loadings at {ZAGREB_STATIONS[0]} (factor by species):")
    print(loadings.round(3).to_string())
    print()

    # --- One Random Forest per pollutant (log target, grid-searched) ------
    print(f"{'pollutant':<10}{'group':<8}{'R2':>8}")
    print("-" * 26)
    for pollutant in POLLUTANTS:
        group = "PAH" if pollutant in PAHS else "metal"
        r2 = rf_per_pollutant(df, pollutant)
        print(f"{pollutant:<10}{group:<8}{r2:>8.3f}")

    # --- TreeSHAP explanation for one pollutant (optional 'shap') ---------
    try:
        from nextaire_tools.models import shap_importance

        station_cols = [c for c in df.columns if c.startswith("station_")]
        X = df[FEATURES + station_cols]
        y = np.log(df["BaP"].to_numpy())
        rf = make_regressor("random_forest", n_estimators=200).fit(X, y)
        imp = shap_importance(rf, X)  # DataFrame columns name the features
        print("\nTop-5 TreeSHAP features for BaP:")
        print(imp.head(5).to_string(index=False))
    except ImportError:
        print("\n  (shap not installed — skipping TreeSHAP; pip install 'nextaire_tools[shap]')")


if __name__ == "__main__":
    main()
