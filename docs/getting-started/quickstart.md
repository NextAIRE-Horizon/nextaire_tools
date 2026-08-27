# Quickstart

This page walks through a complete, runnable air-quality workflow: load an
hourly station file, clean and feature-engineer it with a
[`Pipeline`](../api/preprocessing.md), take a quick look with a plot, and
cross-validate a regressor with a **leakage-free** time-series split. It assumes
you have installed the core package (see [Installation](installation.md)).

The running example is a datetime-indexed, hourly frame with three pollutant
columns — `no2`, `o3`, `pm10` — and we predict `no2` from the others plus
calendar features.

## 1. Load the data

`load_table` reads CSV, Excel, or Parquet and, with `set_time_index=True`, parses
`time_col` into a sorted `DatetimeIndex`:

```python
from nextaire_tools import load_table

df = load_table("station.csv", time_col="timestamp", set_time_index=True)
df.shape
# (720, 3)         # 30 days of hourly no2 / o3 / pm10
df.columns.tolist()
# ['no2', 'o3', 'pm10']
```

A `DatetimeIndex` matters downstream: `MissingValueHandler`'s `"interpolate"`
strategy switches to time-aware interpolation, and `TemporalFeatures` can read
the timestamp straight off the index. See [Loading data](../user-guide/loading-data.md)
for delimiters, sheets, and column subsetting.

## 2. Clean and engineer features with a Pipeline

A [`Pipeline`](../api/preprocessing.md) threads a **single DataFrame** through an
ordered list of steps. Here we impute gaps, tame outliers, add calendar +
cyclical time features, and scale the predictor columns:

```python
from nextaire_tools import Pipeline
from nextaire_tools.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    TemporalFeatures,
    Scaler,
)

pipe = Pipeline([
    MissingValueHandler(strategy="interpolate", limit=6),
    OutlierHandler(columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"),
    TemporalFeatures(
        add=("hour", "dayofweek", "month", "dayofyear", "is_weekend"),
        cyclical=("hour", "dayofweek", "month", "dayofyear"),
    ),
    Scaler(columns=["o3", "pm10"], method="standard"),
])

clean = pipe.fit_transform(df)
clean.shape
# (720, 16)
clean.columns.tolist()
# ['no2', 'o3', 'pm10', 'hour', 'dayofweek', 'month', 'dayofyear',
#  'is_weekend', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
#  'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos']
```

What each step does:

- **`MissingValueHandler(strategy="interpolate", limit=6)`** — bridges gaps of up
  to 6 consecutive hours using time interpolation (because the frame has a
  `DatetimeIndex`).
- **`OutlierHandler(method="iqr", strategy="clip")`** — learns per-column
  `(low, high)` bounds on the training data and clips values to them, so a sensor
  spike is capped rather than deleted.
- **`TemporalFeatures(...)`** — appends the five raw calendar columns plus, for
  each of the four `cyclical` fields, a `sin`/`cos` pair (8 columns).
- **`Scaler(columns=["o3", "pm10"])`** — standardizes the two exogenous
  predictors, leaving the target `no2` in its original units.

Every step is fitted, so you can inspect what it learned:

```python
pipe.named_steps["outlierhandler1"].outlier_fraction_
# 0.0347         # ~3.5% of training rows had at least one clipped value
```

!!! danger "Checkpoint — NaN after interpolate"
    `strategy="interpolate"` with `limit=k` only bridges gaps of **up to `k`**
    consecutive missing values. Leading/trailing NaNs and longer gaps survive —
    and a later `Scaler` will happily propagate them, poisoning your feature
    matrix. Always confirm you are clean before modeling:

    ```python
    int(clean.isna().sum().sum())
    # 0
    ```

    If it is non-zero, append a final `MissingValueHandler(strategy="drop")`
    (rows) or use `limit=None`. See
    [Missing values](../user-guide/missing-values.md).

!!! tip "Checkpoint — clip, don't blindly drop"
    A `pm10` of 500 µg/m³ might be a sensor fault *or* a real pollution episode.
    `strategy="clip"` (or `"flag"`) preserves the row so you can inspect it;
    `strategy="drop"` throws the event away. Note also that
    `method="isolation_forest"` is multivariate and **cannot** be combined with
    `strategy="clip"`. See [Outliers](../user-guide/outliers.md).

## 3. Take a quick look

Every `nextaire_tools.viz` function returns a Matplotlib `Axes` (or an array of them) and
**never** calls `plt.show()` — so it composes into your own figures and saves
cleanly. Pass `save_path=` to write a file, or call `plt.show()` yourself:

```python
import matplotlib.pyplot as plt
from nextaire_tools.viz import plot_seasonality

plot_seasonality(df, column="no2", by="hour")   # mean no2 by hour of day
plt.show()
```

`by=` also accepts `"dayofweek"` and `"month"`. For missingness, distributions,
correlations, and time-series overviews see [Visualization](../user-guide/visualization.md).

## 4. Cross-validate a model — without leakage

Keep the target as a DataFrame column right up to the split, then separate `X`
and `y`. Never shuffle a time series: use a `nextaire_tools` splitter so no future row is
used to predict the past. `BlockingTimeSeriesSplit` cuts the series into
independent blocks, and `gap` drops rows between train and test to emulate a
forecast lead time (here, 24 hours):

```python
from nextaire_tools.models import make_regressor, cross_val_report, BlockingTimeSeriesSplit

X, y = clean.drop(columns="no2"), clean["no2"]
X.shape, y.shape
# ((720, 15), (720,))

cv = BlockingTimeSeriesSplit(n_splits=5, gap=24)
report = cross_val_report(
    make_regressor("random_forest", n_estimators=200),
    X, y,
    cv=cv,
    metrics=["mae", "rmse", "r2", "index_of_agreement", "fac2"],
)
print(report.round(3))
```

`cross_val_report` returns a DataFrame with **one row per fold** plus `mean` and
`std` summary rows (index named `fold`):

```text
        mae   rmse     r2  index_of_agreement   fac2
fold
0     2.151  2.738  0.822               0.951  0.983
1     2.613  3.349  0.840               0.950  0.967
2     3.234  4.179  0.663               0.913  0.983
3     2.690  3.465  0.770               0.939  1.000
4     3.086  3.910  0.724               0.927  0.950
mean  2.755  3.528  0.764               0.936  0.977
std   0.427  0.554  0.072               0.016  0.019
```

!!! warning "Checkpoint — no shuffling, ever"
    `sklearn.model_selection.KFold` and `train_test_split(shuffle=True)` mix
    future and past and will **inflate** these scores. Always use the
    chronological splitters in [`nextaire_tools.models`](../api/models.md)
    (`BlockingTimeSeriesSplit`, `SlidingWindowSplit`, `ExpandingWindowSplit`,
    `TimeSeriesSplit`) and set `gap` to your forecast horizon. See
    [Modeling & cross-validation](../user-guide/modeling.md).

### A single held-out score

For a plain train/test evaluation, `temporal_train_test_split` takes the last
slice as the test set (no shuffle), and `regression_metrics` returns the full
battery of air-quality metrics as a dict:

```python
from nextaire_tools.models import temporal_train_test_split, regression_metrics

train, test = temporal_train_test_split(clean, test_size=0.2, gap=24)
len(train), len(test)
# (552, 144)

model = make_regressor("random_forest", n_estimators=200)
model.fit(train.drop(columns="no2"), train["no2"])
pred = model.predict(test.drop(columns="no2"))

regression_metrics(test["no2"], pred)
# {'mae': 2.652, 'mse': 11.338, 'rmse': 3.367, 'r2': 0.815,
#  'mape': 18.759, 'smape': 16.279, 'bias': 0.218,
#  'pearson_r': 0.903, 'spearman_r': 0.908,
#  'index_of_agreement': 0.947, 'fac2': 0.979}
```

!!! note "The numbers are illustrative"
    The values above come from a synthetic diurnal series and will differ for
    your data. `index_of_agreement` (Willmott's *d*, 0–1, 1 = perfect) and
    `fac2` (fraction of predictions within a factor of two of the observation)
    are atmospheric-science staples; the full list is in the
    [metrics glossary](../reference/metrics.md).

## Next steps

You now have the whole arc: **load → clean → visualize → validate → model**. Go
deeper in the user guide:

- [Key concepts](concepts.md) — the step / pipeline mental model and why it is DataFrame-in / DataFrame-out.
- [Pipelines](../user-guide/pipelines.md) — naming, indexing, and `Pipeline.from_config`.
- [Missing values](../user-guide/missing-values.md) · [Outliers](../user-guide/outliers.md) · [Temporal features](../user-guide/temporal-features.md) · [Scaling](../user-guide/scaling.md)
- [Modeling & cross-validation](../user-guide/modeling.md) and [Deep learning](../user-guide/deep-learning.md)
- [Copernicus extractors](../user-guide/extractors.md) — add ERA5 / CAMS reanalysis features.
