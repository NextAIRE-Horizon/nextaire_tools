# Temporal features

Time *is* the signal in an hourly pollutant series: NO₂ tracks the rush-hour
cycle, O₃ tracks the diurnal photochemistry, and everything drifts with the
seasons. [`TemporalFeatures`](../api/preprocessing.md) turns a timestamp — the
frame's `DatetimeIndex` or a named column — into two complementary families of
predictors: raw **calendar fields** and smooth **cyclical `sin`/`cos`
encodings**.

```python
from nextaire_tools.preprocessing import TemporalFeatures
```

Throughout we use the hourly, datetime-indexed frame from
[Loading data](loading-data.md).

## Two families of features

```python
step = TemporalFeatures(
    add=("hour", "dayofweek", "month", "is_weekend"),
    cyclical=("hour", "dayofweek", "dayofyear"),
)
out = step.fit_transform(df)
```

- **`add`** appends raw integer/boolean calendar columns: `hour`, `dayofweek`,
  `month`, `is_weekend`, ...
- **`cyclical`** appends a `sin`/`cos` pair per field: `hour_sin`, `hour_cos`,
  `dayofweek_sin`, `dayofweek_cos`, `dayofyear_sin`, `dayofyear_cos`.

The defaults are
`add=("hour", "dayofweek", "month", "dayofyear", "is_weekend")` and
`cyclical=("hour", "dayofweek", "month", "dayofyear")`.

### Supported calendar fields (`add`)

`year`, `quarter`, `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`,
`hour`, `minute`, `is_weekend`, `is_month_start`, `is_month_end`, `season`
(`1`=DJF, `2`=MAM, `3`=JJA, `4`=SON), and `is_holiday`.

`dayofweek` follows pandas: Monday is `0`, Sunday is `6`; `is_weekend` is `1` on
Saturday and Sunday.

### Cyclically-encodable fields (`cyclical`)

`hour`, `minute`, `dayofweek`, `month`, `quarter`, `dayofyear`, `weekofyear`,
`day`. Flags and unbounded fields (`is_weekend`, `year`, `season`, ...) cannot be
cyclically encoded.

## Why cyclical encoding matters

!!! danger "Checkpoint — hour 23 is adjacent to hour 0"
    A raw integer `hour` tells a model that 23 and 0 are **23 units apart**, when
    in reality they are one hour apart. The same discontinuity breaks
    `dayofweek` (Sunday → Monday) and `dayofyear` (Dec 31 → Jan 1). Any
    distance- or gradient-based model — kNN, SVR, neural nets, even linear
    models — is misled by that artificial jump.

    The fix is to map each periodic field onto a circle. For a value $v$ with
    base $b$ and period $p$,

    $$\theta = \frac{2\pi\,(v - b)}{p}, \qquad
      v_{\sin} = \sin\theta, \qquad v_{\cos} = \cos\theta.$$

    The `(sin, cos)` pair is continuous across the wrap-around: hour 23 and hour
    0 land next to each other on the unit circle. `TemporalFeatures` does this
    for you via `cyclical=...`.

The base/period pairs are chosen so each field wraps at its natural boundary:

| Field | base $b$ | period $p$ |
| --- | --- | --- |
| `hour` | 0 | 24 |
| `minute` | 0 | 60 |
| `dayofweek` | 0 | 7 |
| `month` | 1 | 12 |
| `quarter` | 1 | 4 |
| `dayofyear` | 1 | 365.25 |
| `weekofyear` | 1 | 52.1775 |
| `day` | 1 | 31 |

Each pair also satisfies $v_{\sin}^2 + v_{\cos}^2 = 1$, so the encoding lives
exactly on the unit circle.

### Seeing the circle

A quick sanity check: plot the two hour components against each other. The 24
hours fall on evenly spaced points around a circle, with 23 sitting right beside
0 — the adjacency a raw integer destroys.

```python
import matplotlib.pyplot as plt

out = TemporalFeatures(add=("hour",), cyclical=("hour",)).fit_transform(df)
plt.scatter(out["hour_cos"], out["hour_sin"], c=out["hour"], cmap="twilight")
plt.gca().set_aspect("equal"); plt.xlabel("hour_cos"); plt.ylabel("hour_sin")
```

### Day-of-week and day-of-year in particular

Two encodings do the heavy lifting for air quality:

- **`dayofweek` ($p=7$)** captures the weekly traffic rhythm — the weekday
  commute peaks versus the quieter weekend — without a Sunday→Monday cliff.
- **`dayofyear` ($p=365.25$)** captures the annual cycle (winter heating, summer
  ozone) smoothly across the New-Year boundary. The `365.25` period keeps leap
  years aligned.

```python
TemporalFeatures(add=(), cyclical=("dayofweek", "dayofyear")).fit_transform(df)
# -> dayofweek_sin, dayofweek_cos, dayofyear_sin, dayofyear_cos
```

A field may appear in `cyclical` without being in `add` (you get only the
`sin`/`cos` pair, as above), and a field in `add` but not `cyclical` yields only
the raw integer.

## Timestamp source: index vs. column

When `time_col` is `None` (default), the frame's `DatetimeIndex` is used:

```python
TemporalFeatures().fit_transform(df)   # df has a DatetimeIndex
```

Pass `time_col` to derive features from a **column** instead. The column is
parsed with `pandas.to_datetime` and **kept** in the output:

```python
import pandas as pd, numpy as np

flat = pd.DataFrame({
    "timestamp": pd.date_range("2024-06-01", periods=48, freq="h"),
    "no2": np.arange(48.0),
})
out = TemporalFeatures(time_col="timestamp",
                       add=("hour",), cyclical=("hour",)).fit_transform(flat)
"timestamp" in out.columns   # True — the source column is preserved
```

!!! warning "You need a timestamp somewhere"
    With neither a `DatetimeIndex` nor a `time_col`, `fit` raises `SchemaError`.
    A `time_col` that is absent raises `ColumnNotFoundError`; one that cannot be
    parsed raises `SchemaError`. Use
    [`load_table(..., time_col=..., set_time_index=True)`](loading-data.md) to
    get a `DatetimeIndex` up front.

## Dropping redundant raw columns

If you keep a field's `sin`/`cos` pair you usually do not also want its raw
integer. Set `drop_raw_cyclical=True` to suppress the raw column for any field
that is cyclically encoded (the pair is still emitted):

```python
step = TemporalFeatures(add=("hour",), cyclical=("hour",), drop_raw_cyclical=True)
out = step.fit_transform(df)
"hour" in out.columns        # False
"hour_sin" in out.columns    # True
```

## Holiday features

`is_holiday` needs a country calendar. Request the field **and** pass an ISO
country code via `holidays_country`; the calendar comes from the optional
[`holidays`](https://pypi.org/project/holidays/) package.

```python
TemporalFeatures(add=("is_holiday",), holidays_country="HR")   # Croatia
```

!!! note "Optional dependency"
    Holiday support requires the `holidays` extra:

    ```bash
    pip install "nextaire_tools[holidays]"
    ```

    Requesting `is_holiday` without `holidays_country` raises
    `ConfigurationError`; requesting it without the package installed raises a
    `MissingDependencyError` at `fit` time.

## Naming the outputs

`prefix` is prepended to **every** generated column name — useful when you build
features from more than one timestamp source and need to disambiguate:

```python
TemporalFeatures(add=("hour",), cyclical=("hour",), prefix="obs_")
# -> obs_hour, obs_hour_sin, obs_hour_cos
```

After `fit`, the newly generated names (in emission order) are on
`feature_names_out_`, and `get_feature_names_out()` returns the **full** output
schema — the original columns followed by the new ones:

```python
step = TemporalFeatures(add=("hour",), cyclical=("hour",)).fit(df)
step.feature_names_out_             # ['hour', 'hour_sin', 'hour_cos']
list(step.get_feature_names_out())  # ['no2', 'o3', 'pm10', 'hour', 'hour_sin', 'hour_cos']
```

Emission order is deterministic: the raw `add` columns first (in the order you
listed them, skipping any suppressed by `drop_raw_cyclical`), then each
`cyclical` field's `_sin` immediately followed by its `_cos`.

!!! example "A typical feature set for hourly pollutants"
    ```python
    step = TemporalFeatures(
        add=("hour", "dayofweek", "month", "is_weekend"),
        cyclical=("hour", "dayofweek", "dayofyear"),
        drop_raw_cyclical=False,
    )
    features = step.fit_transform(df)
    ```

    This keeps interpretable raw columns *and* the smooth encodings that models
    actually learn best from.

## See also

- [Pipelines](pipelines.md) — put `TemporalFeatures` after cleaning and before
  scaling.
- [Scaling](scaling.md) — the `sin`/`cos` pairs are already in $[-1, 1]$; you
  usually scale only the raw pollutant columns.
- [Visualization](visualization.md) — [`plot_seasonality`](../api/viz.md)
  cross-checks the cycles these features encode.
- [API reference: `TemporalFeatures`](../api/preprocessing.md).
