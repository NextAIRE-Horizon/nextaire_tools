# Outliers

Sensor spikes, negative concentrations, and stuck values all show up as
outliers in air-quality data — but so do **real pollution episodes**.
[`OutlierHandler`](../api/preprocessing.md) detects anomalies with a choice of
statistical rules or a multivariate isolation forest, then treats them by
clipping, dropping, masking to `NaN`, or flagging.

```python
from nextaire_tools.preprocessing import OutlierHandler
```

We reuse the hourly frame from [Loading data](loading-data.md) with two obvious
spikes injected:

```python
df.iloc[50, df.columns.get_loc("pm10")] = 500.0     # implausible PM10 spike
df.iloc[300, df.columns.get_loc("no2")] = -200.0    # impossible negative NO2
```

## Detection methods

Set the method with `method`. The four **bound methods** learn a per-column
`(low, high)` interval during `fit`; `isolation_forest` learns a multivariate
detector instead.

### `iqr` (default)

Flags values outside the inter-quartile fence
$[\,Q_1 - k\cdot\mathrm{IQR},\; Q_3 + k\cdot\mathrm{IQR}\,]$, where
$\mathrm{IQR}=Q_3-Q_1$ and $k$ is `iqr_factor` (default `1.5`). Distribution-free
and a sensible default for skewed pollutant data.

```python
OutlierHandler(method="iqr", iqr_factor=1.5)
```

### `zscore`

Flags values more than `z_threshold` standard deviations from the mean:
$[\,\mu - z\sigma,\; \mu + z\sigma\,]$ (default `z_threshold=3.0`). Assumes
roughly normal data and is itself sensitive to the very outliers it looks for.

### `modified_zscore`

A robust variant built on the median absolute deviation
$\mathrm{MAD}=\operatorname{median}(|x-\operatorname{median}(x)|)$. The bounds are

$$\operatorname{median}(x)\ \pm\ \frac{t\cdot \mathrm{MAD}}{0.6745},$$

where $t$ is `mad_threshold` (default `3.5`) and `0.6745` rescales the MAD to
match the standard deviation of a normal distribution. Prefer this over `zscore`
when the data already contains spikes.

### `quantile`

Clips to empirical quantiles, `quantiles=(0.01, 0.99)` by default — i.e. treat
the most extreme 1% on each tail as outliers. Purely rank-based:

```python
OutlierHandler(method="quantile", quantiles=(0.005, 0.995))
```

### `isolation_forest`

A multivariate, **row-level** detector
([`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)):
it scores an entire row across the selected columns, catching joint anomalies
that no single-column rule would (e.g. a plausible `no2` at an implausible
combination with `o3`). Set the expected `contamination` and a `random_state`
for reproducibility.

```python
OutlierHandler(
    columns=["no2", "o3", "pm10"],
    method="isolation_forest",
    strategy="drop",
    contamination=0.05,
    random_state=0,
)
```

!!! note "Missing values are never flagged"
    For the bound methods, comparisons against `NaN` are `False`, so missing
    cells are left for [`MissingValueHandler`](missing-values.md) to deal with.
    The isolation forest cannot score rows that contain a `NaN` in the selected
    columns and treats them as non-outliers (it needs at least one complete row
    to fit at all).

## Treatment strategies

`strategy` decides what happens to the detected outliers:

| `strategy` | Effect | Row count | Notes |
| --- | --- | --- | --- |
| `"clip"` (default) | Clip each column to its learned bounds | unchanged | Bound methods only |
| `"drop"` | Drop rows containing an outlier | shrinks | Loses real data |
| `"nan"` | Replace outlying cells with `NaN` | unchanged | Feed to a missing-value step next |
| `"flag"` | Append an integer `"is_outlier"` column | unchanged | Keeps everything; lets the model decide |

```python
OutlierHandler(columns=["pm10"], method="iqr", strategy="clip").fit_transform(df)
OutlierHandler(columns=["pm10"], method="iqr", strategy="nan").fit_transform(df)
OutlierHandler(columns=["pm10"], method="iqr", strategy="flag").fit_transform(df)
```

With `strategy="flag"`, `get_feature_names_out()` includes the extra
`"is_outlier"` column. With `strategy="nan"`, chain a missing-value step
afterwards to fill the holes you just created.

## Don't delete the signal you came to study

!!! danger "Checkpoint — outliers vs. real events"
    A PM10 reading of 500 µg/m³ might be a faulty sensor **or** a genuine
    wildfire-smoke episode — the number alone cannot tell you which. Blindly
    dropping "outliers" can erase exactly the extreme-pollution events an
    air-quality model exists to predict.

    - Prefer **`"flag"`** or **`"clip"`** over `"drop"`. Flagging keeps every
      row and lets the model learn from the event; clipping tames the magnitude
      without deleting the timestamp.
    - **Inspect before you treat** with [`plot_outliers`](#visual-inspection).
    - Physically impossible values (negative concentrations) are safe to treat;
      merely *large* values deserve a second look.

## Combining with isolation forest — a guard rail

!!! warning "`isolation_forest` + `clip` raises `ConfigurationError`"
    Clipping needs a per-column `(low, high)` interval, but the isolation forest
    is inherently multivariate and learns no such bounds — so the combination is
    rejected at `fit` time:

    ```python
    from nextaire_tools.exceptions import ConfigurationError

    try:
        OutlierHandler(method="isolation_forest", strategy="clip").fit(df)
    except ConfigurationError as e:
        print(e)   # use 'drop', 'nan', or 'flag'
    ```

    Selecting a non-numeric column raises `SchemaError`, and a malformed
    `quantiles` pair raises `ConfigurationError`.

## Inspecting the fit

```python
step = OutlierHandler(columns=["no2", "pm10"], method="iqr").fit(df)

step.bounds_             # {'no2': (low, high), 'pm10': (low, high)} — empty for isolation_forest
step.n_outliers_         # int: number of outlier rows on the training data
step.outlier_fraction_   # float in [0, 1]
step.detector_           # the fitted IsolationForest, or None for bound methods
```

## Visual inspection

[`plot_outliers`](../api/viz.md) draws a single series and highlights the
out-of-bounds points. It can read the interval straight from a **fitted**
handler (duck-typed on `bounds_`), so you see exactly what the step will treat:

```python
from nextaire_tools.viz import plot_boxplots, plot_outliers

step = OutlierHandler(columns=["pm10"], method="iqr").fit(df)
plot_outliers(df, "pm10", handler=step)      # bounds drawn from the fitted step
plot_outliers(df, "no2", bounds=(0, 200))    # or pass explicit bounds
```

For a quick multi-column overview, [`plot_boxplots`](../api/viz.md) shows the
spread and fliers of every numeric column at once:

```python
plot_boxplots(df, columns=["no2", "o3", "pm10"])
```

!!! example "A defensible outlier step"
    ```python
    # Clip absurd magnitudes but keep every timestamp, then look at the result.
    step = OutlierHandler(columns=["no2", "o3", "pm10"],
                          method="modified_zscore", strategy="clip").fit(df)
    print(step.outlier_fraction_)
    plot_outliers(df, "pm10", handler=step)
    clean = step.transform(df)
    ```

## See also

- [Missing values](missing-values.md) — pair with `strategy="nan"` to detect
  then impute.
- [Scaling](scaling.md) — always scale **after** outlier handling, or a single
  spike distorts the scale for every other point.
- [Visualization](visualization.md) and
  [API reference: `OutlierHandler`](../api/preprocessing.md).
