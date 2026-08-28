# nextaire_tools

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22149642.svg)](https://doi.org/10.5281/zenodo.22149642)
[![Docs](https://readthedocs.org/projects/nextaire-tools/badge/?version=latest)](https://nextaire-tools.readthedocs.io/en/latest/)

Preprocessing, feature engineering, Copernicus data extraction, visualization,
and ML/DL modeling for air-quality time series.

`nextaire_tools` collects the steps a typical air-quality study repeats by hand — loading
a table, handling missing values and outliers, building calendar features,
pulling ERA5/CAMS reanalysis, plotting, and fitting a model with a correct
time-series split — into a set of composable, tested, scikit-learn-compatible
building blocks. It implements the methods from three peer-reviewed studies (see
[Reproducing the papers](#reproducing-the-papers)), so a study can be rebuilt from
documented, versioned code rather than one-off notebooks.

## Contents

- [Installation](#installation)
- [Main features](#main-features)
- [Example data](#example-data)
- [Quickstart](#quickstart)
- [Tutorials & notebooks](#tutorials--notebooks)
- [Reproducing the papers](#reproducing-the-papers)
- [Documentation](#documentation)
- [Citing nextaire_tools](#citing-nextaire_tools)

## Installation

```bash
pip install nextaire_tools                 # core: IO, preprocessing, viz, sklearn models
pip install "nextaire_tools[deep]"         # + PyTorch (MLP / LSTM / CNN)
pip install "nextaire_tools[extract]"      # + Copernicus (cdsapi, xarray, cfgrib)
pip install "nextaire_tools[forecast]"     # + Prophet (and the Prophet→RF hybrid)
pip install "nextaire_tools[boost]"        # + XGBoost
pip install "nextaire_tools[shap]"         # + TreeSHAP interpretation
pip install "nextaire_tools[geo]"          # + geospatial land-use features (geopandas, osmnx)
pip install "nextaire_tools[notebooks]"    # + JupyterLab to run the example notebooks
pip install "nextaire_tools[all]"          # everything
```

## Main features

Everything is a `DataFrame`-in / `DataFrame`-out step, so column names and the
`DatetimeIndex` survive from raw table to fitted model.

### Data IO

`load_table` reads CSV, Excel, and Parquet into a datetime-indexed `DataFrame`
(with DMS/decimal coordinate handling for station files); `save_table` writes them
back. One entry point for messy inputs.

### Cleaning

Scikit-learn transformers for the parts every study gets slightly differently:

- `MissingValueHandler` — drop, statistical / directional fill, time-aware
  interpolation, or multivariate **iterative** imputation.
- `OutlierHandler` — IQR, z-score, modified z-score, quantile bounds, a
  multivariate Isolation Forest, or **`rolling_sigma`** time-local winsorisation
  that removes short spikes without deleting real pollution events.

### Feature engineering

- `TemporalFeatures` — calendar fields plus sine/cosine encodings (hour 23 is
  adjacent to hour 0).
- `WindDecomposer` — wind direction → x/y components, and speed from u/v.
- `LagFeatures` — causal lags and rolling aggregates (e.g. 12-hour median).
- `CorrelationFilter` — drop collinear features above a threshold.
- `Scaler` and `Pipeline` / `make_pipeline` compose it all reproducibly.

### Copernicus extraction

`ERA5Extractor`, `CAMSExtractor`, and `ERA5LandExtractor` download from the
Climate/Atmosphere Data Stores and sample the grid at your monitoring stations;
`load_stations` parses the coordinate files.

### Visualization

A colorblind-safe, light/dark-aware Matplotlib theme with EDA
(`plot_missingness`, `plot_correlation`, `plot_seasonality`, `plot_timeseries`),
outlier inspection, and evaluation (`plot_predictions`, `plot_residuals`,
`plot_feature_importance`) figures.

### Modeling & validation

- Leakage-free splitters: `BlockingTimeSeriesSplit`, `SlidingWindowSplit`,
  `ExpandingWindowSplit`, `temporal_train_test_split`.
- `make_regressor` — Random Forest, gradient boosting, decision tree, linear
  family, KNN, SVR, and optional XGBoost by name.
- PyTorch `MLPRegressor` / `LSTMRegressor` / `CNNRegressor`, a `ProphetForecaster`,
  and a `HybridProphetRegressor` (Prophet forecasts as features for a regressor).
- `NMFApportionment` for NMF source apportionment.

### Metrics & interpretation

`regression_metrics` reports the scores atmospheric scientists actually use —
MAE/RMSE/R², index of agreement, FAC2, and IQR-normalized nMAE/nRMSE plus WAPE —
and `cross_val_report` tabulates them per fold. `permutation_importance_report`
and `shap_importance` (TreeSHAP) explain a fitted model.

### Module map

| Subpackage | Key names |
|------------|-----------|
| `nextaire_tools.io` | `load_table`, `save_table` |
| `nextaire_tools.preprocessing` | `MissingValueHandler`, `OutlierHandler`, `TemporalFeatures`, `WindDecomposer`, `LagFeatures`, `CorrelationFilter`, `Scaler`, `Pipeline` |
| `nextaire_tools.extractors` | `ERA5Extractor`, `CAMSExtractor`, `ERA5LandExtractor`, `load_stations` |
| `nextaire_tools.viz` | `plot_missingness`, `plot_seasonality`, `plot_predictions`, … |
| `nextaire_tools.models` | `make_regressor`, `BlockingTimeSeriesSplit`, `cross_val_report`, `regression_metrics`, `LSTMRegressor`, `HybridProphetRegressor`, `NMFApportionment`, `shap_importance` |

## Example data

No downloads are needed to try `nextaire_tools`. The generators in
[`reproductions/_synthetic.py`](reproductions/_synthetic.py) build data with the
same columns, frequency, and structure as the papers' real datasets:

- `make_graz_hourly()` / `make_graz_multistation()` — hourly pollutants
  (NO/NO₂/O₃/PM10) plus ground meteorology and ERA5-style reanalysis, with
  realistic diurnal/seasonal cycles, gaps, and spikes.
- `make_zagreb_daily()` — daily PM10-bound PAHs and metals with meteorology,
  traffic and heating proxies, and station labels.

```python
import sys; sys.path.insert(0, "reproductions")
from _synthetic import make_graz_hourly

df = make_graz_hourly(n_days=180, seed=0)   # hourly, datetime-indexed DataFrame
```

For real data, replace the generator with `nextaire_tools.load_table("station.csv",
time_col="timestamp", set_time_index=True)`.

## Quickstart

```python
import nextaire_tools
from nextaire_tools import load_table, Pipeline
from nextaire_tools.preprocessing import MissingValueHandler, OutlierHandler, TemporalFeatures, Scaler

# 1. Load anything (CSV / Excel / Parquet) with a datetime index
df = load_table("station.csv", time_col="timestamp", set_time_index=True)

# 2. Build a reproducible cleaning + feature pipeline
pipe = Pipeline([
    MissingValueHandler(strategy="interpolate", limit=3),
    OutlierHandler(columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"),
    TemporalFeatures(
        add=("hour", "dayofweek", "month", "is_weekend"),
        cyclical=("hour", "dayofweek", "dayofyear"),   # sin/cos encodings
    ),
    Scaler(method="standard"),
])
clean = pipe.fit_transform(df)
```

### Explore before you model

```python
from nextaire_tools.viz import plot_missingness, plot_correlation, plot_seasonality

plot_missingness(df)
plot_correlation(df, cluster=True)
plot_seasonality(df, column="o3", by="hour")
```

### Fit a model with a correct time-series split

```python
from nextaire_tools.models import make_regressor, cross_val_report, BlockingTimeSeriesSplit

target = "no2"
X = clean.drop(columns=[target])
y = clean[target]

model = make_regressor("random_forest", n_estimators=400)
report = cross_val_report(model, X, y, cv=BlockingTimeSeriesSplit(n_splits=5))
print(report)   # per-fold MAE / RMSE / R² / nMAE / nRMSE / IoA / FAC2 + mean & std
```

### Deep learning (optional `[deep]`)

```python
from nextaire_tools.models import LSTMRegressor

lstm = LSTMRegressor(window=24, hidden_size=64, epochs=50)
lstm.fit(X.values, y.values)
y_hat = lstm.predict(X.values)
```

### Pull ERA5 meteorology at your stations (optional `[extract]`)

```python
from nextaire_tools.extractors import ERA5Extractor, load_stations

stations = load_stations("data/Coordinates.xlsx")   # handles DMS or decimal degrees
era5 = ERA5Extractor(output_dir="data/era5")
frames = era5.extract_to_frames(
    stations=stations,
    variables=["2m_temperature", "10m_u_component_of_wind", "boundary_layer_height"],
    area=[49.1, 9.53, 46.3, 17.16],   # N, W, S, E
    start="2024-01-01", end="2024-01-31",
    save_dir="data/era5",
)
```

## Tutorials & notebooks

Runnable Jupyter notebooks in [`notebooks/`](notebooks/) work on the synthetic
example data — no network required — and are committed with their executed
outputs, so they render on GitHub and re-run top to bottom.

| Notebook | What it covers |
|----------|----------------|
| [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Load/save, EDA plots, a cleaning + feature `Pipeline`, and a Random Forest with a leakage-free split. |
| [`02_preprocessing_and_features.ipynb`](notebooks/02_preprocessing_and_features.ipynb) | Every preprocessing step in turn, then composed into one pipeline. |
| [`03_reproduce_papers.ipynb`](notebooks/03_reproduce_papers.ipynb) | Compact interactive versions of the three paper recipes. |
| [`04_deep_learning_and_forecasting.ipynb`](notebooks/04_deep_learning_and_forecasting.ipynb) | PyTorch MLP/LSTM/CNN regressors and Prophet / hybrid Prophet+RF forecasting. |

```bash
pip install "nextaire_tools[notebooks]"
jupyter lab                              # open a notebook, or run headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_quickstart.ipynb
```

There is also a plain-script walkthrough in
[`examples/end_to_end.py`](examples/end_to_end.py) and a prose tutorial in the
[documentation](https://nextaire-tools.readthedocs.io/en/latest/).

## Reproducing the papers

`nextaire_tools` packages the methodology of three peer-reviewed air-quality ML studies.
Each has a runnable recipe in [`reproductions/`](reproductions/) that rebuilds its
data construction, preprocessing, cross-validation, models, and metrics — offline,
on synthetic data of the same shape, degrading gracefully when an optional
dependency is missing. Swap the synthetic generator for `load_table(...)` of the
real data to reproduce the published numbers.

```bash
pip install "nextaire_tools[all]"
python reproductions/paper1_petric2024_aaqr.py            # Petrić et al. 2024 (AAQR)
python reproductions/paper2_jimenez2024_multitarget.py   # Jiménez-Navarro et al. 2024 (Results in Eng.)
python reproductions/paper3_racic2026_source_apportionment.py  # Račić et al. 2026 (Atmos. Env. X)
```

See [`reproductions/README.md`](reproductions/README.md) for the full paper →
API map, and [`papers/README.md`](papers/README.md) for citations and data
sources.

## Documentation

Full documentation — user guide, API reference, and tutorials — lives at
**[nextaire-tools.readthedocs.io](https://nextaire-tools.readthedocs.io/en/latest/)** and is built from the
Markdown sources in [`docs/`](docs/).

Build it locally:

```bash
pip install "nextaire_tools[docs]"
mkdocs serve   # http://127.0.0.1:8000
```

## Citing nextaire_tools

If you use `nextaire_tools` in academic work, please cite the relevant methodological
paper(s) it is based on:

> Petrić, V., Hussain, H., Časni, K., et al. (2024). *Ensemble Machine Learning,
> Deep Learning, and Time Series Forecasting: Improving Prediction Accuracy for
> Hourly Concentrations of Ambient Air Pollutants.* Aerosol and Air Quality Research,
> 24, 230317. doi:10.4209/aaqr.230317

> Jiménez-Navarro, M. J., Lovrić, M., Kecorius, S., Nyarko, E. K.,
> Martínez-Ballesteros, M. (2024). *Explainable deep learning on multi-target time
> series forecasting: An air pollution use case.* Results in Engineering, 24,
> 103290. doi:10.1016/j.rineng.2024.103290

> Račić, N., Ružičić, S., Petrić, V., et al. (2026). *Assessment of contributors to
> airborne PAHs and heavy metals in PM₁₀ using temporal, spatial, traffic and
> heating data in explainable machine learning models.* Atmospheric Environment: X,
> 29, 100413. doi:10.1016/j.aeaoa.2026.100413

You can also cite the software itself:

> Petrić, V. (2026). *nextaire_tools: Air-quality time-series preprocessing,
> extraction, and ML/DL* [Software]. Zenodo. doi:10.5281/zenodo.22149642

See [`docs/about/citation.md`](docs/about/citation.md) for BibTeX entries.

## License

`nextaire_tools` is released under the [MIT License](LICENSE).
