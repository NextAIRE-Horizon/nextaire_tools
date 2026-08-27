# Modeling & cross-validation

`nextaire_tools.models` provides the pieces of a *correct* supervised workflow for
air-quality time series: leakage-free cross-validation splitters, a battery of
regression metrics tuned for atmospheric-science evaluation, a one-line
scikit-learn regressor factory, and a `cross_val_report` that ties them
together. This page covers the classical (scikit-learn) path; the neural models
live in [Deep learning](deep-learning.md).

See the [`nextaire_tools.models` API reference](../api/models.md) for full signatures.

!!! danger "Checkpoint: never shuffle a time series"
    The single most common way to get an *inflated, meaningless* score on a time
    series is to shuffle it. `sklearn.model_selection.KFold` and
    `train_test_split(shuffle=True)` mix future rows into the training fold; on a
    temporally correlated pollutant series that leaks the answer and the model
    looks far better than it will in production.

    **Always split chronologically.** Use the nextaire_tools splitters below (or
    `temporal_train_test_split`), all of which guarantee `train.max() <
    test.min()`, and use their `gap` parameter to emulate a real forecast lead
    time. If you must use a sklearn helper, it is
    `sklearn.model_selection.TimeSeriesSplit` (re-exported here) — never `KFold`.

## The splitters

Four splitters share the scikit-learn `split(X)` / `get_n_splits(X)` protocol,
so any of them drops straight into `cross_val_report` (and sklearn's own
`cross_validate`). In the diagrams below, `#` is a training sample, `=` a test
sample, `·` an unused sample, and time runs left → right.

### `BlockingTimeSeriesSplit(n_splits=5, gap=0)`

Divides the series into `n_splits` contiguous, **non-overlapping** blocks; within
each block the earlier half trains and the later half tests. Unlike
`TimeSeriesSplit`, folds never share samples, so each block is evaluated on its
own footing.

```text
BlockingTimeSeriesSplit(n_splits=3)          30 samples
fold 0:  #####=====  ··········  ··········
fold 1:  ··········  #####=====  ··········
fold 2:  ··········  ··········  #####=====
```

```python
from nextaire_tools.models import BlockingTimeSeriesSplit

cv = BlockingTimeSeriesSplit(n_splits=5)          # requires n_splits >= 2
for train_idx, test_idx in cv.split(X):
    assert train_idx.max() < test_idx.min()       # no leakage
```

### `TimeSeriesSplit` (scikit-learn, re-exported)

Expanding-window folds that share a common origin: the training set grows and
each successive test block follows it. Re-exported from scikit-learn so you can
get every splitter from one namespace.

```text
TimeSeriesSplit(n_splits=3)
fold 0:  #####=====················
fold 1:  ##########=====···········
fold 2:  ###############=====······
```

```python
from nextaire_tools.models import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=24)          # gap also supported by sklearn
```

### `SlidingWindowSplit(train_size, test_size, *, step=None, gap=0)`

A **fixed-width** training window that rolls forward by `step` each fold (default
`step = test_size`, giving contiguous test windows). Old samples fall out as new
ones enter, so this measures how well a model tracks a possibly non-stationary
series using only recent history.

```text
SlidingWindowSplit(train_size=10, test_size=5)
fold 0:  ##########=====···············
fold 1:  ·····##########=====··········
fold 2:  ··········##########=====·····
```

```python
from nextaire_tools.models import SlidingWindowSplit

cv = SlidingWindowSplit(train_size=24 * 30, test_size=24 * 7)   # 30-day train, 7-day test
cv.get_n_splits(X)      # needs X to know how many windows fit
```

### `ExpandingWindowSplit(initial_train_size, test_size, *, step=None, gap=0)`

A **growing** training window anchored at the first sample, with a fixed-width
test window rolling just after it. Mirrors production, where a model is
periodically retrained on all history so far.

```text
ExpandingWindowSplit(initial_train_size=10, test_size=5)
fold 0:  ##########=====···············
fold 1:  ###############=====··········
fold 2:  ####################=====·····
```

```python
from nextaire_tools.models import ExpandingWindowSplit

cv = ExpandingWindowSplit(initial_train_size=24 * 60, test_size=24 * 7)
```

### The `gap` parameter

Every splitter accepts `gap`: the number of samples **dropped between the train
and test blocks**. This emulates a forecast lead time — if you predict 24 hours
ahead, the last 24 hours before the test block must not be in training, or the
model peeks at data it would not have at inference time.

```text
SlidingWindowSplit(train_size=10, test_size=5, gap=3)
fold 0:  ##########···=====············
         └─ train ─┘ gap └test┘
```

!!! tip "Match `gap` to your horizon"
    Set `gap` to your forecast horizon in *samples*. For an hourly series and a
    24-hour-ahead forecast, `gap=24`. This is the honest way to score a
    forecaster: it never trains on the hours it is asked to predict across.

### `temporal_train_test_split`

For a single hold-out (not cross-validation), split off the most recent tail
without shuffling. The two returned objects keep the type of the input
(`DataFrame`, `Series`, array, list, or tuple).

```python
from nextaire_tools.models import temporal_train_test_split

train, test = temporal_train_test_split(df, test_size=0.2)   # last 20% is test
train, test = temporal_train_test_split(df, test_size=168, gap=24)  # explicit count + 24h gap
```

`test_size` is a fraction in `(0, 1)` or an integer count; `gap` discards the
samples immediately before the test set. Order is always preserved
(`train.index.max() < test.index.min()`).

## Choosing a regressor

`make_regressor(name="random_forest", **params)` builds any supported
scikit-learn estimator from a short name, so you can select a model by
configuration. `**params` are forwarded to the constructor and override
nextaire_tools's defaults.

```python
from nextaire_tools.models import make_regressor, list_regressors

list_regressors()
# ['elasticnet', 'extra_trees', 'gradient_boosting', 'hist_gradient_boosting',
#  'knn', 'lasso', 'linear', 'mlp', 'random_forest', 'ridge', 'svr']

rf = make_regressor("random_forest", n_estimators=300)
ridge = make_regressor("ridge", alpha=2.0)
```

| Name | Estimator |
| --- | --- |
| `linear` | `LinearRegression` |
| `ridge` | `Ridge` |
| `lasso` | `Lasso` |
| `elasticnet` | `ElasticNet` |
| `random_forest` | `RandomForestRegressor` |
| `extra_trees` | `ExtraTreesRegressor` |
| `gradient_boosting` | `GradientBoostingRegressor` |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor` |
| `svr` | `SVR` |
| `knn` | `KNeighborsRegressor` |
| `mlp` | `MLPRegressor` (scikit-learn) |

!!! note "Reproducible tree ensembles, and the `mlp` name"
    The tree-based models (`random_forest`, `extra_trees`, `gradient_boosting`,
    `hist_gradient_boosting`) default to `random_state=0` for reproducibility;
    pass your own `random_state=` to override. Note that `make_regressor("mlp")`
    returns scikit-learn's CPU `MLPRegressor` — **not** the PyTorch
    `nextaire_tools.models.MLPRegressor` from the [deep-learning](deep-learning.md) extra,
    which is a distinct class.

## Metrics

`regression_metrics(y_true, y_pred, *, metrics=None)` returns a dict of
point-forecast scores. Inputs are flattened and paired element-wise, and any
pair with a NaN in either array is dropped before scoring — so gappy observation
series (or the leading NaNs produced by windowed deep models) are handled
transparently.

```python
from nextaire_tools.models import regression_metrics

m = regression_metrics(y_test, y_pred)
m["rmse"], m["r2"], m["index_of_agreement"], m["fac2"]

# Restrict to a subset:
regression_metrics(y_test, y_pred, metrics=["mae", "rmse", "index_of_agreement"])
```

The full set (`METRIC_NAMES` order): `mae`, `mse`, `rmse`, `r2`, `mape`,
`smape`, `bias`, `pearson_r`, `spearman_r`, `index_of_agreement`, `fac2`. Two
are specific to atmospheric-science evaluation and worth defining precisely:

!!! example "Index of agreement (Willmott's *d*) and FAC2"
    **Index of agreement** measures how well predictions track observations
    around the *observed* mean, on a 0–1 scale where 1 is perfect:

    $$
    d = 1 - \frac{\sum_i (p_i - o_i)^2}{\sum_i \big(|p_i - \bar{o}| + |o_i - \bar{o}|\big)^2}
    $$

    where $p_i$ are predictions, $o_i$ observations, and $\bar{o}$ the mean of
    the observations. Unlike R², it is bounded in $[0, 1]$ and penalizes both
    bias and variance error.

    **FAC2** (factor-of-two) is the fraction of predictions within a factor of
    two of the observation — the share of points with
    $0.5 \le p_i / o_i \le 2$. It is a robust, interpretable "how often are we in
    the right ballpark?" score widely reported in dispersion-model evaluation.

    Both `mape` and `fac2` ignore observations equal to zero (the relative error
    is undefined there). See the [Metrics glossary](../reference/metrics.md) for
    the rest.

## `cross_val_report`

`cross_val_report(model, X, y, *, cv, metrics=None, clone_estimator=True)` runs
an estimator over a splitter and tabulates the held-out metrics. Each fold fits
on its training rows and scores on its held-out rows; a fresh `sklearn.base.clone`
is trained per fold (unless `clone_estimator=False`) so folds never share fitted
state.

It returns a `DataFrame` with **one row per fold** (index `0, 1, …`), a column
per metric, and two appended summary rows, `"mean"` and `"std"`, computed across
the folds. The index is named `"fold"`.

```python
from nextaire_tools.models import cross_val_report, make_regressor, BlockingTimeSeriesSplit

report = cross_val_report(
    make_regressor("random_forest"),
    X, y,
    cv=BlockingTimeSeriesSplit(n_splits=5),
    metrics=["rmse", "r2", "index_of_agreement", "fac2"],
)
print(report)
#         rmse        r2  index_of_agreement      fac2
# fold
# 0     7.213   0.612               0.842     0.930
# 1     6.984   0.641               0.861     0.945
# ...
# mean  7.050   0.628               0.851     0.938
# std   0.201   0.019               0.011     0.008
```

## A full example

Build features and target from a cleaned frame, cross-validate leakage-free, and
inspect the fit. The key convention is that the **target stays a column of the
same frame** as the features, so any row-dropping preprocessing keeps them
aligned (see the checkpoint below).

```python
import nextaire_tools
from nextaire_tools import load_table, Pipeline
from nextaire_tools.preprocessing import MissingValueHandler, TemporalFeatures, Scaler
from nextaire_tools.models import (
    make_regressor, cross_val_report, temporal_train_test_split,
    BlockingTimeSeriesSplit,
)
from nextaire_tools.viz import plot_predictions, plot_feature_importance

# 1. Load and clean (target 'no2' remains a column throughout).
df = load_table("station.csv", time_col="timestamp", set_time_index=True)
clean = Pipeline([
    MissingValueHandler(strategy="interpolate", limit=6),
    MissingValueHandler(strategy="drop"),        # drop any NaN the interpolation left
    TemporalFeatures(cyclical=("hour", "dayofweek", "dayofyear")),
    Scaler(columns=["no2", "o3", "pm10"], method="standard"),
]).fit_transform(df)

# 2. Split features/target AFTER cleaning, so rows stay aligned.
target = "no2"
X = clean.drop(columns=[target])
y = clean[target]

# 3. Leakage-free cross-validation.
report = cross_val_report(
    make_regressor("random_forest", n_estimators=300),
    X, y,
    cv=BlockingTimeSeriesSplit(n_splits=5),
)
print(report.loc[["mean", "std"]])

# 4. Fit on a chronological train split and inspect the held-out tail.
X_train, X_test = temporal_train_test_split(X, test_size=0.2, gap=24)
y_train, y_test = temporal_train_test_split(y, test_size=0.2, gap=24)

model = make_regressor("random_forest", n_estimators=300).fit(X_train, y_train)
y_pred = model.predict(X_test)

plot_predictions(y_test, y_pred, kind="timeseries", index=y_test.index)
plot_feature_importance(model, top_n=15)
```

!!! danger "Checkpoint: keep the target aligned with the features"
    A row-dropping step (`MissingValueHandler(strategy="drop")`,
    `OutlierHandler(strategy="drop")`) changes which rows survive. If your
    features and target are separate objects when that happens, they silently go
    **out of alignment** and every subsequent score is wrong.

    The nextaire_tools convention avoids this: keep the target as a **column** of the
    frame and split it off (`X = clean.drop(columns=target); y = clean[target]`)
    *after* all row-dropping preprocessing. Do **not** put a row-dropping step
    inside a scikit-learn `Pipeline` that carries a separate `y` — sklearn only
    reindexes `X`. See [Pipelines](pipelines.md) for how nextaire_tools steps move rows
    together.

!!! warning "Interpolate then drop"
    `MissingValueHandler(strategy="interpolate", limit=k)` only bridges gaps up
    to `k` steps; leading/trailing NaNs and longer gaps remain. A later `Scaler`
    (or a model) will happily propagate those NaNs into your scores. Either chain
    a final `strategy="drop"` (as above) or use `limit=None`. See
    [Missing values](missing-values.md).

## See also

- [Deep learning](deep-learning.md) — PyTorch MLP/LSTM/CNN behind the same
  `fit`/`predict` API, and how they flow through `cross_val_report`.
- [Pipelines](pipelines.md) — assembling the preprocessing that produces `X`/`y`.
- [Visualization](visualization.md) — `plot_predictions`, `plot_residuals`,
  `plot_feature_importance`.
- [Metrics glossary](../reference/metrics.md).
- [`nextaire_tools.models` API reference](../api/models.md).
