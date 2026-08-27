# Scaling

Many estimators — kNN, SVR, neural networks, and any distance- or
gradient-based model — expect features on comparable scales. [`Scaler`](../api/preprocessing.md)
wraps the scikit-learn scalers but, unlike them, returns a `DataFrame`: it scales
only the selected numeric columns in place within a copy, leaving every other
column, dtype, and the (datetime) index untouched, and it can invert the
transform.

```python
from nextaire_tools.preprocessing import Scaler
```

We reuse the hourly `no2`/`o3`/`pm10` frame from [Loading data](loading-data.md).

## Choosing a method

| `method` | Transform | Output range | When to use |
| --- | --- | --- | --- |
| `"standard"` (default) | $(x-\mu)/\sigma$ | ~zero mean, unit variance | General default for roughly symmetric data |
| `"minmax"` | $(x-\min)/(\max-\min)$ | $[0, 1]$ | Bounded inputs; **sensitive to outliers** |
| `"robust"` | $(x-\operatorname{median})/\mathrm{IQR}$ | centered, unbounded | Skewed data or residual outliers |
| `"maxabs"` | $x / \max|x|$ | $[-1, 1]$ | Preserve sign and sparsity; no centering |

```python
Scaler(method="standard").fit_transform(df)
Scaler(method="robust").fit_transform(df)
```

The fitted scikit-learn scaler is available on `scaler_` if you need its learned
attributes (e.g. `scaler_.mean_` for `"standard"`).

## Scaling only some columns

With `columns=None` (default) every numeric column is scaled. Pass `columns` to
restrict scaling — every other column and the index pass through unchanged:

```python
step = Scaler(columns=["no2"], method="minmax")
out = step.fit_transform(df)

out["no2"].min(), out["no2"].max()   # ~0.0, ~1.0  — scaled
out["pm10"].equals(df["pm10"])       # True         — untouched
```

This is exactly what you want after
[`TemporalFeatures`](temporal-features.md): the cyclical `sin`/`cos` columns are
already in $[-1, 1]$, so scale only the raw pollutant columns.

## Round-tripping with `inverse_transform`

`Scaler` remembers enough to map scaled values back to their original units,
which is essential when you scale a **target** and want predictions in real
concentrations:

```python
step = Scaler(columns=["no2", "o3"], method="standard")
scaled = step.fit_transform(df)
restored = step.inverse_transform(scaled)

import numpy as np
np.allclose(restored["no2"], df["no2"])   # True
```

`inverse_transform` needs the fitted columns to be present in its input;
otherwise it raises `ColumnNotFoundError`. All non-fitted columns and the index
are returned unchanged.

## Order matters: scale last

!!! warning "Scale *after* outlier handling"
    `"standard"` and `"minmax"` are computed from the mean/min/max, so a single
    spike drags the whole column's scale with it — every other point gets
    squashed into a sliver of the range. Run
    [`OutlierHandler`](outliers.md) (clip, drop, or nan) **before** `Scaler`, or
    use `method="robust"`, which relies on the median and IQR and shrugs off
    extreme values. In a pipeline, scaling is normally the **last** numeric step.

## Missing values pass straight through

!!! danger "Checkpoint — `Scaler` does not fill `NaN`"
    `Scaler` scales; it never imputes. Any `NaN` remaining in a column stays
    `NaN` in the output — the scaler computes its statistics ignoring missing
    values and leaves the gaps in place. That is easy to overlook after a
    **bounded** interpolation, which can leave leading, trailing, or
    longer-than-`limit` gaps behind (see the
    [NaN-after-interpolate checkpoint](missing-values.md#the-interpolation-trap)).
    Those `NaN`s then reach your estimator and break `fit`.

    Always resolve missing values *before* scaling and confirm it:

    ```python
    from nextaire_tools.preprocessing import MissingValueHandler, Scaler, Pipeline

    pipe = Pipeline([
        MissingValueHandler(strategy="interpolate", limit=3),
        MissingValueHandler(strategy="drop"),   # clear residual NaNs
        Scaler(method="standard"),
    ])
    clean = pipe.fit_transform(df)
    assert clean.isna().to_numpy().sum() == 0
    ```

!!! example "Fit on train, apply to test"
    Because `Scaler` learns its statistics in `fit`, split first and reuse the
    fitted scaler so test-set statistics never leak into training:

    ```python
    from nextaire_tools.models import temporal_train_test_split

    train, test = temporal_train_test_split(df, test_size=0.2)
    scaler = Scaler(columns=["no2", "o3", "pm10"]).fit(train)
    train_s = scaler.transform(train)
    test_s = scaler.transform(test)   # same scale as train
    ```

## See also

- [Missing values](missing-values.md) — must run before scaling.
- [Outliers](outliers.md) — must run before scaling (or use `robust`).
- [Pipelines](pipelines.md) — the canonical clean → feature → scale order.
- [API reference: `Scaler`](../api/preprocessing.md).
