# Changelog

All notable changes to **nextaire_tools** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Paper reproductions** — runnable, offline recipes in `reproductions/` that
  rebuild the methodology of the three studies `nextaire_tools` is based on (Petrić et al.
  2024; Jiménez-Navarro et al. 2024; Račić et al. 2026). See `papers/README.md`.
- **Example notebooks** — `notebooks/` with a quickstart, a preprocessing/feature
  tour, interactive paper reproductions, and a deep-learning + Prophet forecasting
  tour; each runs on synthetic data and is committed with executed outputs. Install
  with `pip install "nextaire_tools[notebooks]"`.
- **Preprocessing**:
  - `OutlierHandler` gains a `rolling_sigma` method — time-local winsorisation on
    a centred rolling window (`window`, `sigma`), for spike removal.
  - `MissingValueHandler` gains an `iterative` strategy — multivariate
    (round-robin) imputation via scikit-learn's `IterativeImputer`, with a
    configurable `estimator` (e.g. Bayesian ridge).
  - `WindDecomposer` — wind direction → sine/cosine (x/y) and speed from u/v
    components.
  - `LagFeatures` — causal lag and rolling-aggregate (e.g. 12-hour median)
    features.
  - `CorrelationFilter` — drop features above a pairwise-correlation threshold.
- **Models**:
  - `HybridProphetRegressor` / `ProphetFeatures` — Prophet forecasts as features
    for a downstream regressor (the "hybrid" model).
  - `NMFApportionment` — rank-k NMF factor analysis for source apportionment.
  - `permutation_importance_report`, `tree_shap_values`, `shap_importance` —
    model-interpretation helpers.
  - `make_regressor` registers `decision_tree` and (optional) `xgboost`.
- **Metrics** — `regression_metrics` adds `wape`, `nmae`, and `nrmse`
  (IQR-normalised).
- **Optional extras** — `nextaire_tools[boost]` (XGBoost) and `nextaire_tools[shap]` (TreeSHAP).

## [0.1.0] — 2026-07-05

Initial public release.

### Added

- **IO** — `load_table` / `save_table` for CSV, Excel, and Parquet with optional
  datetime-index handling.
- **Preprocessing** (scikit-learn–compatible, DataFrame-in/DataFrame-out steps):
  - `MissingValueHandler` — drop / impute / interpolate missing values, missingness
    indicators, and column dropping by missing fraction.
  - `OutlierHandler` — IQR, z-score, modified z-score, quantile, and Isolation-Forest
    detection with clip / drop / NaN / flag strategies.
  - `TemporalFeatures` — calendar features plus cyclical (sine/cosine) encodings for
    hour, day-of-week, month, and day-of-year.
  - `Scaler` — standard / min-max / robust / max-abs scaling with `inverse_transform`.
  - `Pipeline` / `make_pipeline` — compose steps; build from config.
- **Extractors** for the Copernicus data stores:
  - `ERA5Extractor` (Climate Data Store), `CAMSExtractor` (Atmosphere Data Store),
    `ERA5LandExtractor` (ERA5-Land) with nearest-neighbour point sampling at stations.
  - `load_stations`, `dms_to_dd`, and GRIB sampling helpers.
- **Visualization** — a colorblind-safe, light/dark-aware plotting theme with EDA,
  outlier, and model-evaluation figures.
- **Models** — time-series cross-validation splitters, air-quality regression metrics
  (incl. index of agreement and FAC2), a scikit-learn regressor factory, PyTorch
  deep-learning regressors (MLP / LSTM / CNN), and a Prophet forecasting wrapper.
- Full MkDocs documentation site, typed API (`py.typed`), and a test suite.

[Unreleased]: https://github.com/NextAIRE-Horizon/nextaire_tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NextAIRE-Horizon/nextaire_tools/releases/tag/v0.1.0
