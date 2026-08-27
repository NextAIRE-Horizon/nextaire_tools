# Missing values

Observational air-quality data is full of gaps: sensors go offline, calibration
windows blank out readings, and telemetry drops. [`MissingValueHandler`](../api/preprocessing.md)
is a DataFrame-in / DataFrame-out step that detects those gaps and either
removes them, imputes them, or fills them in a time-aware way — while never
mutating your input frame.

```python
from nextaire_tools.preprocessing import MissingValueHandler
```

We use the hourly `no2`/`o3`/`pm10` frame from [Loading data](loading-data.md),
with a few injected gaps:

```python
import numpy as np

df.iloc[5:8, df.columns.get_loc("no2")] = np.nan   # a 3-hour gap
df.iloc[100, df.columns.get_loc("o3")] = np.nan
```

## Choosing a strategy

Every strategy is selected with the `strategy` argument. The table below is the
quick-reference; the sections after it explain the trade-offs.

| `strategy` | What it does | Typical use |
| --- | --- | --- |
| `"drop"` | Drop rows with a `NaN` in **any** selected column | Small gap fraction; you can afford to lose rows |
| `"mean"` / `"median"` | Impute with the learned column mean / median | Numeric columns; `median` is robust to skew (e.g. `pm10`) |
| `"most_frequent"` | Impute with the column mode | Categorical / discrete columns |
| `"constant"` | Impute with `fill_value` | A meaningful sentinel (e.g. `0.0`) |
| `"ffill"` / `"bfill"` | Carry the last / next value forward / backward | Slowly-varying signals; short gaps only (see `limit`) |
| `"interpolate"` | Time-aware interpolation between known points | Smooth hourly series — the usual default for pollutants |

The default is `strategy="drop"`.

### Dropping rows

```python
out = MissingValueHandler(strategy="drop").fit_transform(df)
```

A row is dropped if **any** of the selected columns is missing in it. Restrict
the check to specific columns with `columns=...` so a gap in a column you do not
care about does not cost you a row:

```python
# Only drop rows where no2 or o3 is missing; ignore gaps elsewhere.
MissingValueHandler(columns=["no2", "o3"], strategy="drop").fit_transform(df)
```

### Statistical imputation

`mean` and `median` learn the fill value from the numeric columns during `fit`
and reuse it in `transform` — so a train/test split is never contaminated by
test-set statistics. `most_frequent` uses the mode; `constant` uses `fill_value`
(which is required for that strategy):

```python
MissingValueHandler(strategy="median").fit_transform(df)
MissingValueHandler(strategy="constant", fill_value=0.0).fit_transform(df)
```

The learned values are exposed on `statistics_` (see
[Inspecting the fit](#inspecting-the-fit)).

### Directional filling and interpolation

`ffill`/`bfill` and `interpolate` honour the `limit` argument — the maximum
number of **consecutive** missing values to bridge. For an hourly series,
interpolation is usually the most faithful choice:

```python
MissingValueHandler(strategy="interpolate", limit=3).fit_transform(df)
```

!!! note "`interpolate` is time-aware on a `DatetimeIndex`"
    When the frame has a `DatetimeIndex`, interpolation uses
    `method="time"`, which weights by the actual time distance between samples —
    correct even for irregularly spaced timestamps. On a non-datetime index it
    falls back to `method="linear"`. Only **numeric** columns are interpolated.

## The interpolation trap

!!! danger "Checkpoint — NaN survives a bounded interpolation"
    `MissingValueHandler(strategy="interpolate", limit=k)` only bridges gaps of
    **up to `k` consecutive** missing values, and interpolation cannot
    extrapolate. That leaves three kinds of `NaN` behind:

    - **Leading** `NaN`s before the first real observation.
    - **Trailing** `NaN`s after the last real observation.
    - Anything **beyond `limit`** inside a long gap.

    Those residual `NaN`s pass straight through a later
    [`Scaler`](scaling.md) — a `NaN` in, a `NaN` out — and will then break most
    estimators at `fit` time. The failure is silent until training.

    **Fixes**, depending on intent:

    ```python
    from nextaire_tools.preprocessing import MissingValueHandler, Pipeline

    # Option A: interpolate interior gaps, then drop whatever remains.
    Pipeline([
        MissingValueHandler(strategy="interpolate", limit=3),
        MissingValueHandler(strategy="drop"),        # clears the edges/long gaps
    ])

    # Option B: bridge everything (no cap) — only if that is defensible.
    MissingValueHandler(strategy="interpolate", limit=None)
    ```

    Verify with `out.isna().sum()` before scaling or modeling.

## Dropping unusable columns

`column_missing_threshold` drops any selected column whose missing fraction
**strictly exceeds** the threshold, before imputation runs. This is handy when a
sensor is dead for most of the record:

```python
step = MissingValueHandler(strategy="median", column_missing_threshold=0.5)
out = step.fit_transform(df)          # columns >50% empty are removed
step.dropped_columns_                 # e.g. ['broken_sensor']
```

## Recording *where* data was missing

Set `add_indicator=True` to append an integer `"<col>__missing"` column (1 where
the original value was absent, 0 otherwise) for every selected column that had at
least one gap. The indicator is computed **before** imputation, so a model can
learn from the pattern of absence itself:

```python
step = MissingValueHandler(columns=["no2"], strategy="mean", add_indicator=True)
out = step.fit_transform(df)
[c for c in out.columns if "missing" in str(c)]   # ['no2__missing']
```

!!! tip "Pair indicators with imputation, not dropping"
    Indicator columns are most useful with an imputing strategy. Combined with
    `strategy="drop"`, the rows that would carry a `1` are removed anyway, so the
    indicator is almost all zeros.

## Inspecting the fit

After `fit`, the step exposes what it learned:

```python
step = MissingValueHandler(strategy="median").fit(df)

step.missing_fraction_    # Series: fraction missing per selected column
step.statistics_          # dict: learned fill value per column (imputers only)
step.dropped_columns_     # list: columns removed by column_missing_threshold
step.indicator_columns_   # list: names of the indicator columns transform adds
```

`get_feature_names_out()` reports the resulting column names (inputs minus
dropped columns, plus any indicator columns).

## Visualizing missingness

Before choosing a strategy, look at the gaps. [`plot_missingness`](../api/viz.md)
draws either a per-column bar chart or a rows-by-columns matrix:

```python
from nextaire_tools.viz import plot_missingness

plot_missingness(df, kind="bar")      # missing fraction per column
plot_missingness(df, kind="matrix")   # where the gaps sit in time
```

The bar view answers "which columns are unreliable?"; the matrix view reveals
whether gaps are scattered or clustered into long outages — which is exactly the
distinction that decides between interpolation and dropping.

## See also

- [Outliers](outliers.md) — the other half of cleaning; run it before imputing
  so spikes do not poison the mean/median.
- [Scaling](scaling.md) — why residual `NaN`s matter downstream.
- [Pipelines](pipelines.md) — chain a bounded interpolation with a final drop.
- [API reference: `MissingValueHandler`](../api/preprocessing.md).
