# End-to-end workflow

This tutorial walks a single hourly air-quality dataset all the way from a raw
station CSV to a cross-validated model and a set of publication-ready figures.
It ties together every part of `nextaire_tools` — [IO](../api/io.md),
[preprocessing](../api/preprocessing.md), [visualization](../api/viz.md), and
[models](../api/models.md) — and calls out the non-obvious pitfalls as
**checkpoints** along the way.

Our running example is a datetime-indexed frame of hourly pollutant columns
(`no2`, `o3`, `pm10`) for one monitoring station. The goal is a **one-hour-ahead
forecast of NO₂**.

!!! note "What you need"
    The core install (`pip install nextaire_tools`) covers everything except the last two
    optional sections. The deep-learning model needs the `deep` extra
    (`pip install 'nextaire_tools[deep]'`) and the meteorology appendix needs the
    `extract` extra (`pip install 'nextaire_tools[extract]'`).

---

## 1. Load the raw station data

`nextaire_tools.load_table` reads CSV, Excel, or Parquet and, given a timestamp column,
returns a frame indexed by a sorted `DatetimeIndex`. A datetime index is what
unlocks time-aware behavior downstream (interpolation by time, calendar
features, seasonality plots), so set it up front.

```python
import nextaire_tools

df = nextaire_tools.load_table(
    "zagreb_station.csv",
    time_col="timestamp",
    set_time_index=True,
)
df.head()
```

```
                          no2        o3      pm10
timestamp
2021-01-01 00:00:00     22.4      38.1      18.9
2021-01-01 01:00:00     19.8      41.0      21.3
2021-01-01 02:00:00      NaN      42.7      17.6
2021-01-01 03:00:00     17.1      44.2       NaN
...
```

!!! tip "One reader for every format"
    The format is inferred from the extension, and extra keyword arguments flow
    straight to the underlying pandas reader. Loading a column subset from
    Parquet pushes the projection down to the reader:
    `load_table("obs.parquet", columns=["timestamp", "no2", "o3"])`. See the
    [Loading data guide](../user-guide/loading-data.md).

---

## 2. Explore before you touch anything

Exploratory plots answer the first questions of any new series: what is missing,
how values are distributed, how the pollutants co-vary, and what daily/seasonal
structure they carry. Every function in [`nextaire_tools.viz`](../api/viz.md) returns a
Matplotlib `Axes` (it never calls `plt.show()`), so charts compose and save
cleanly.

```python
from nextaire_tools.viz import (
    set_style, plot_missingness, plot_distributions,
    plot_correlation, plot_timeseries, plot_seasonality,
)

set_style("notebook")  # apply the shared nextaire_tools look once

plot_missingness(df, kind="matrix", save_path="fig/missingness.png")
plot_distributions(df, ncols=3, save_path="fig/distributions.png")
plot_correlation(df, method="spearman", annot=True, save_path="fig/corr.png")
plot_timeseries(df, ["no2", "o3", "pm10"], save_path="fig/series.png")
plot_seasonality(df, "no2", by="hour", save_path="fig/no2_by_hour.png")
```

The `plot_seasonality` panel for NO₂ by hour typically shows the twin
rush-hour peaks — a strong daily cycle that motivates the cyclical time features
in step 4. The correlation heatmap tells you which pollutants carry mutual
information (useful predictors), and `plot_missingness` reveals whether gaps are
scattered (imputable) or arrive in long blocks (not safely imputable).

!!! warning "Read the gaps before choosing a strategy"
    A missingness *matrix* distinguishes short, scattered gaps from long
    contiguous blocks. That distinction decides step 3: interpolation is safe
    across a scattered gap of a few hours, but fabricates data across a
    multi-day outage.

---

## 3. Build the preprocessing pipeline

`nextaire_tools.preprocessing.Pipeline` threads a **single DataFrame** through an ordered
list of steps. Unlike a scikit-learn `Pipeline`, it never separates `X` from
`y`, so steps that change the row count (dropping gaps or outliers) keep every
column aligned. We split preprocessing into two pipelines around the point where
the target is defined.

First, the **row-preserving cleaning** pipeline: impute short gaps, then tame
outliers by clipping.

```python
from nextaire_tools.preprocessing import (
    Pipeline, MissingValueHandler, OutlierHandler, TemporalFeatures, Scaler,
)

clean_pipeline = Pipeline([
    ("impute",   MissingValueHandler(columns=["no2", "o3", "pm10"],
                                     strategy="interpolate", limit=6)),
    ("outliers", OutlierHandler(columns=["no2", "o3", "pm10"],
                                method="iqr", strategy="clip")),
])
clean = clean_pipeline.fit_transform(df)
```

!!! danger "Checkpoint — interpolation leaves NaNs behind"
    `MissingValueHandler(strategy="interpolate", limit=6)` only bridges gaps of
    **up to six** consecutive hours. Leading/trailing gaps and longer outages
    stay `NaN`, and a later `Scaler` will happily propagate that `NaN` into every
    model input. Either raise/drop `limit`, or (as we do in step 4) end with an
    explicit `dropna()` so no `NaN` reaches the model. Verify with
    `clean[["no2","o3","pm10"]].isna().sum()`.

!!! danger "Checkpoint — outliers may be real pollution events"
    A PM₁₀ spike can be a genuine episode (fireworks, wildfire smoke, a dust
    event), not an instrument fault. Don't reflexively `strategy="drop"`. Prefer
    `"clip"` or `"flag"` and *inspect* before deciding — see
    `plot_outliers(df, "pm10", handler=fitted_handler)` in the
    [outliers guide](../user-guide/outliers.md). Note that `method="isolation_forest"`
    is multivariate (row-level) and **cannot** be combined with
    `strategy="clip"`.

You can inspect what the outlier handler learned:

```python
handler = clean_pipeline.named_steps["outliers"]   # fitted by fit_transform above
handler.bounds_          # {'no2': (low, high), 'o3': (...), 'pm10': (...)}
handler.outlier_fraction_
```

---

## 4. Define the target and engineer features

Now define the supervised target as **next-hour NO₂**, keeping it in the frame
as a plain column (`no2_next`) so it rides along through the remaining steps and
stays row-aligned. Then add calendar and cyclical time features and scale the
pollutant inputs.

```python
clean = clean.assign(no2_next=clean["no2"].shift(-1))

featurize = Pipeline([
    ("calendar", TemporalFeatures(
        add=("hour", "dayofweek", "month", "is_weekend"),
        cyclical=("hour", "dayofweek", "month"),
    )),
    ("scale", Scaler(columns=["no2", "o3", "pm10"], method="standard")),
])
frame = featurize.fit_transform(clean)
```

`TemporalFeatures` emits the raw integer fields you name in `add` plus a
`<field>_sin` / `<field>_cos` pair for each field in `cyclical`. The `Scaler`
touches only its `columns`, leaving `no2_next` (and the new calendar columns)
untouched.

!!! danger "Checkpoint — encode cyclical time as sin/cos"
    Hour 23 is one step from hour 00, but the raw integers 23 and 0 are 23 apart.
    A model reads that jump as a huge discontinuity every midnight. `cyclical=(...)`
    maps each periodic field onto the unit circle so the wrap-around is smooth.
    Keep the raw integer only if a model can use it (trees can); set
    `drop_raw_cyclical=True` to emit the sin/cos pair alone.

!!! danger "Checkpoint — keep the target as a column, then drop NaN once"
    Because `no2_next` is a column, the single `dropna()` below removes the
    trailing shifted row (and any residual imputation `NaN`) from features **and**
    target together — they can never fall out of sync. This is the nextaire_tools
    convention: do row-dropping on one frame, *then* separate `X` and `y`. Do
    **not** feed a row-dropping step and a separate `y` into a scikit-learn
    `Pipeline`.

```python
frame = frame.dropna()
feature_cols = [c for c in frame.columns if c != "no2_next"]
X = frame[feature_cols]      # DataFrame of features
y = frame["no2_next"]        # Series, original NO₂ units
```

!!! warning "Scaling and leakage"
    Fitting the `Scaler` on the whole series lets test-set statistics leak into
    training. The tree models below are scale-invariant, so scaling is harmless
    (you may omit it). For scale-sensitive models (linear, SVR, k-NN, MLP), fit
    the scaler on the **training slice only** — split first (step 6), then
    `Scaler().fit_transform(X_train)` / `.transform(X_test)`.

---

## 5. Cross-validate — without leaking the future

Ordinary `KFold` and `train_test_split(shuffle=True)` shuffle rows, which drops
future observations into the training fold and **inflates every score**. Use a
[nextaire_tools splitter](../api/models.md) that respects chronological order.
`BlockingTimeSeriesSplit` carves the series into independent, non-overlapping
blocks; the `gap` drops samples between train and test to emulate the forecast
lead time.

!!! danger "Checkpoint — never shuffle a time series"
    Any splitter or `train_test_split` with `shuffle=True` leaks the future.
    Match `gap` to your horizon: we forecast one hour ahead, so `gap=1` removes
    the single step that a real forecaster would not yet have observed.

`cross_val_report` fits a fresh clone of the estimator per fold and tabulates the
held-out metrics, with `mean` and `std` summary rows. We compare a random forest
against histogram gradient boosting.

```python
from nextaire_tools.models import make_regressor, cross_val_report, BlockingTimeSeriesSplit

cv = BlockingTimeSeriesSplit(n_splits=4, gap=1)
scores = ["mae", "rmse", "r2", "index_of_agreement", "fac2"]

for name in ["random_forest", "hist_gradient_boosting"]:
    model = make_regressor(name)
    report = cross_val_report(model, X, y, cv=cv, metrics=scores)
    print(name)
    print(report.round(3))
```

An abbreviated report looks like this (values are illustrative):

```
random_forest
        mae   rmse     r2  index_of_agreement   fac2
fold
0     2.71   3.55  0.783               0.935  0.972
1     2.80   3.69  0.774               0.931  0.976
2     2.74   3.61  0.781               0.934  0.975
3     2.83   3.73  0.779               0.932  0.976
mean  2.77   3.65  0.779               0.933  0.975
std   0.05   0.08  0.004               0.002  0.002
```

`make_regressor` accepts any of `linear`, `ridge`, `lasso`, `elasticnet`,
`random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`,
`svr`, `knn`, `mlp` (see `list_regressors()`), and forwards keyword arguments to
the estimator (e.g. `make_regressor("random_forest", n_estimators=300)`). The
tree ensembles default to `random_state=0` for reproducibility.

!!! tip "Which metrics to trust for air quality"
    Report **index of agreement** (Willmott's *d*) and **FAC2** alongside RMSE
    and R². They are the atmospheric-science standard because *d* is bounded and
    sensitive to both bias and scatter, and FAC2 states plainly what fraction of
    predictions land within a factor of two. See the
    [metrics glossary](../reference/metrics.md).

---

## 6. Fit, predict, and diagnose on a held-out tail

For the final diagnostics, carve a chronological holdout with
`temporal_train_test_split` (no shuffling; `gap=1` again for the one-hour
horizon), fit on the past, and predict on the future tail.

```python
from nextaire_tools.models import temporal_train_test_split, regression_metrics
from nextaire_tools.viz import plot_predictions, plot_residuals, plot_feature_importance

X_train, X_test = temporal_train_test_split(X, test_size=0.2, gap=1)
y_train, y_test = temporal_train_test_split(y, test_size=0.2, gap=1)

rf = make_regressor("random_forest", n_estimators=300)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

regression_metrics(y_test, y_pred, metrics=["mae", "rmse", "r2",
                                            "index_of_agreement", "fac2"])
# {'mae': 2.89, 'rmse': 3.61, 'r2': 0.79, 'index_of_agreement': 0.94, 'fac2': 0.97}
```

Now the three evaluation charts:

=== "Predicted vs observed"

    ```python
    plot_predictions(y_test, y_pred, kind="scatter",
                     save_path="fig/pred_scatter.png")
    ```

    A scatter with the dashed 1:1 line and the R² in the title. Points hugging
    the diagonal are good; a systematic offset from it signals bias.

=== "Time series"

    ```python
    plot_predictions(y_test, y_pred, kind="timeseries", index=y_test.index,
                     save_path="fig/pred_series.png")
    ```

    Observed and predicted overlaid in time — the view for spotting missed peaks
    and lagged responses.

=== "Residuals"

    ```python
    plot_residuals(y_test, y_pred, save_path="fig/residuals.png")
    ```

    Two panels: residuals vs predicted (look for structure or a non-zero mean)
    and the residual distribution.

Because `X` is a DataFrame, the fitted estimator carries `feature_names_in_`, so
`plot_feature_importance` labels the bars automatically:

```python
plot_feature_importance(rf, top_n=12, save_path="fig/importance.png")
```

You will usually see the autoregressive `no2` term and `hour_sin`/`hour_cos`
near the top — exactly the daily structure the seasonality plot foreshadowed.

---

## 7. (Optional) A recurrent model

For a sequence model, `LSTMRegressor` (the `deep` extra) windows the series
internally. It standardizes inputs and targets on its own, so no external
`Scaler` is required. Feed it NumPy arrays.

```python
from nextaire_tools.models import LSTMRegressor

lstm = LSTMRegressor(window=24, hidden_size=64, epochs=50, random_state=0)
lstm.fit(X_train.to_numpy(), y_train.to_numpy())
lstm_pred = lstm.predict(X_test.to_numpy())
```

!!! danger "Checkpoint — the first `window` predictions are NaN"
    `LSTMRegressor` (and `CNNRegressor`) return an array of length `len(X)`, but
    the first `window` rows have no full look-back window and come back as `NaN`.
    `regression_metrics` drops `NaN` pairs automatically, so scoring still works:

    ```python
    regression_metrics(y_test, lstm_pred, metrics=["mae", "index_of_agreement"])
    ```

    Do not paper over those `NaN`s with `fillna(0)` before scoring — that would
    fabricate perfect-looking or terrible-looking errors at the series start.

See the [deep-learning guide](../user-guide/deep-learning.md) for `MLPRegressor`,
`CNNRegressor`, and the lower-level `make_sequences` helper.

---

## 8. (Optional) Attach ERA5 meteorology

Wind, temperature, and boundary-layer height are strong pollutant predictors.
`ERA5Extractor` (the `extract` extra) downloads ERA5 reanalysis from the
Copernicus Climate Data Store and samples it at your station coordinates.

```python
from nextaire_tools.extractors import load_stations, ERA5Extractor

stations = load_stations("stations.xlsx")   # -> station_name, station_lon, station_lat

era5 = ERA5Extractor()
frames = era5.extract_to_frames(
    stations=stations,
    area=[46.0, 15.5, 45.5, 16.5],   # [North, West, South, East]
    start="2021-01-01",
    end="2021-12-31",
    save_dir="data/era5",
)
met = frames["Zagreb"]               # one timestamp-indexed frame per station
```

Join the (hourly) meteorology onto your pollutant frame by timestamp before
step 3, so the new columns flow through the same cleaning and feature steps:

```python
df = df.join(met, how="left")
```

!!! danger "Checkpoint — coordinates, area order, and credentials"
    - **Area order is `[North, West, South, East]`** — not the GeoJSON `[W, S, E, N]`.
      `CopernicusExtractor.expand_area(stations, margin=0.5)` computes a correct box.
    - Station coordinates may be **DMS strings**; `load_stations` parses both DMS
      and decimal degrees. Sampling is **nearest-neighbour** to the reanalysis grid.
    - You need CDS credentials (a Personal Access Token in `~/.cdsapirc` or the
      `CDSAPI_KEY` environment variable). **CAMS** uses a *different* store (ADS)
      with its own token; **ERA5-Land** has no `product_type` key; ERA5 is hourly
      while CAMS EAC4 is 3-hourly.

!!! warning "Run large downloads from a script"
    A year of hourly ERA5 can take many minutes to hours in the CDS queue. Run
    extraction from a standalone script or a background job, not inline in a
    notebook you are waiting on. See the
    [Copernicus data sources reference](../reference/data-sources.md) and the
    [extractors guide](../user-guide/extractors.md).

---

## Where to go next

- Read the [Metrics glossary](../reference/metrics.md) for the exact formulas and
  the "when is this misleading" notes.
- Browse the [API reference](../api/index.md) for full signatures.
