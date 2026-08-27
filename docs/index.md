---
hide:
  - navigation
---

# nextaire_tools

Preprocessing, feature engineering, Copernicus data extraction, visualization,
and ML/DL modeling for air-quality time series.

`nextaire_tools` collects the steps a typical air-quality study repeats by hand — loading
a table, handling missing values and outliers, building calendar features,
pulling ERA5/CAMS reanalysis, plotting, and fitting a model with a correct
time-series split — into a set of composable, tested, scikit-learn-compatible
building blocks. It implements the methods used in the peer-reviewed studies
documented under [Reproducing the papers](reference/reproducing-papers.md).

- **Load data** — `load_table` reads CSV, Excel, and Parquet into a
  datetime-indexed `DataFrame`. See [Loading data](user-guide/loading-data.md).
- **Clean and engineer** — missing-value handling, outlier detection, cyclical
  temporal features, and scaling, each a reusable transformer. See
  [Pipelines](user-guide/pipelines.md).
- **Copernicus extraction** — download ERA5, CAMS, and ERA5-Land and sample the
  grid at monitoring stations. See [Extractors](user-guide/extractors.md).
- **Model** — leakage-free time-series cross-validation, air-quality metrics, and
  classical plus deep (LSTM/CNN/MLP) models. See [Modeling](user-guide/modeling.md).

## Installation

```bash
pip install nextaire_tools                 # core
pip install "nextaire_tools[deep]"         # + PyTorch (MLP / LSTM / CNN)
pip install "nextaire_tools[extract]"      # + Copernicus (cdsapi, xarray, cfgrib)
pip install "nextaire_tools[all]"          # everything
```

See [Installation](getting-started/installation.md) for all extras and the note on
the distribution name.

## Example

```python
import nextaire_tools
from nextaire_tools import load_table, Pipeline
from nextaire_tools.preprocessing import MissingValueHandler, OutlierHandler, TemporalFeatures, Scaler

df = load_table("station.csv", time_col="timestamp", set_time_index=True)

pipe = Pipeline([
    MissingValueHandler(strategy="interpolate"),
    OutlierHandler(columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"),
    TemporalFeatures(cyclical=("hour", "dayofweek", "dayofyear")),   # sin/cos encodings
    Scaler(method="standard"),
])
clean = pipe.fit_transform(df)
```

Then explore, cross-validate, and model:

```python
from nextaire_tools.viz import plot_seasonality
from nextaire_tools.models import make_regressor, cross_val_report, BlockingTimeSeriesSplit

plot_seasonality(df, column="o3", by="hour")

X, y = clean.drop(columns="no2"), clean["no2"]
report = cross_val_report(make_regressor("random_forest"), X, y,
                          cv=BlockingTimeSeriesSplit(n_splits=5))
```

New here? Start with the [Quickstart](getting-started/quickstart.md), then read
[Key concepts](getting-started/concepts.md) for the step/pipeline model.

## Citing

If you use `nextaire_tools`, please [cite the relevant paper(s)](about/citation.md).
