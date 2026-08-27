# Visualization

`nextaire_tools.viz` is a small charting layer for air-quality time series: a single,
colorblind-safe Matplotlib theme plus three families of figures — exploratory
data analysis (EDA), outlier inspection, and model evaluation. Every function
takes a `DataFrame` (or arrays), returns the Matplotlib `Axes` it drew on, and
**never calls `plt.show()`** — so it composes cleanly in notebooks, scripts, and
saved reports.

The running example is an hourly, datetime-indexed frame of pollutant columns
`no2`, `o3`, `pm10` (see [Loading data](loading-data.md)):

```python
import nextaire_tools
from nextaire_tools import load_table

df = load_table("station.csv", time_col="timestamp", set_time_index=True)
df.head()
#                       no2     o3   pm10
# timestamp
# 2024-01-01 00:00:00  20.4   55.1   18.7
# 2024-01-01 01:00:00  24.1   52.8   21.0
# ...
```

For the full list of parameters and return types, see the
[`nextaire_tools.viz` API reference](../api/viz.md).

## The theme

Import `set_style` once at the top of a session to apply the nextaire_tools look
globally. It is idempotent — calling it repeatedly leaves the style unchanged —
and every plotting function calls it lazily the first time it runs, so charts
are consistent even if you forget.

```python
from nextaire_tools.viz import set_style

set_style()              # default: seaborn "notebook" context, grid on
set_style("talk")        # larger fonts/marks for slides
set_style("paper", grid=False)
```

`set_style(context="notebook", *, grid=True)` sets a single cohesive design:
an off-white figure surface, hidden top/right spines, a recessive hairline
grid, thin marks, and a fixed eight-color categorical cycle. `context` is the
seaborn scaling (`"paper"`, `"notebook"`, `"talk"`, `"poster"`).

### Color tokens

The palette and colormaps are exported so you can reuse them in your own charts
and stay on-brand:

```python
from nextaire_tools.viz import PALETTE, INK, SEQUENTIAL_CMAP, DIVERGING_CMAP

PALETTE[0]          # '#2a78d6' — first categorical hue
len(PALETTE)        # 8
INK["grid"]         # hairline gridline color
```

| Token | Purpose |
| --- | --- |
| `PALETTE` | Eight-color categorical cycle, assigned to series by position |
| `INK` | Non-series ink: `primary`, `secondary`, `muted`, `grid`, `baseline`, `surface` |
| `SEQUENTIAL_CMAP` | Single-hue blue ramp for magnitudes (e.g. the missingness map) |
| `DIVERGING_CMAP` | Blue↔red ramp with a neutral midpoint for signed values (e.g. correlation) |

!!! tip "Color follows the entity, not its rank"
    Series are colored from `PALETTE` in **column order**, never sorted by
    value. The palette hues are chosen to stay distinguishable for viewers with
    color-vision deficiency, and they read on both light and dark figure
    backgrounds. `PALETTE` holds exactly eight colors and is never cycled beyond
    that — pass at most eight series to one axes, or switch to small multiples
    (see the design rule below).

If you want the nextaire_tools look only inside a `with` block — leaving global
`rcParams` untouched afterwards — use the `nextaire_tools_style` context manager:

```python
from nextaire_tools.viz.style import nextaire_tools_style

with nextaire_tools_style("talk"):
    ax = df["no2"].plot()
```

!!! note "Design rule: small multiples, never a second y-axis"
    nextaire_tools charts never draw a secondary (twin) y-axis — dual axes make the
    relative scale of two series arbitrary and easy to misread. When series
    live on different scales, draw a **grid of small multiples** instead (one
    panel per variable). `plot_distributions` and `plot_residuals` already
    return such a grid, and `plot_timeseries` logs a warning once you exceed the
    eight-color palette, nudging you toward small multiples or aggregating minor
    series into an "Other" line.

### `ax=`, `figsize=`, and `save_path=`

Every function shares the same three plumbing arguments:

- `ax=` — draw onto an existing `Axes` (or array of axes) instead of creating a
  new figure. This is how you compose small multiples by hand.
- `figsize=` — size of the new figure, used only when `ax is None`.
- `save_path=` — when given, the figure is written (tight bounding box, 150 dpi)
  and the `Axes` is returned; nothing is displayed.

```python
import matplotlib.pyplot as plt
from nextaire_tools.viz import plot_timeseries, plot_seasonality

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_timeseries(df, ["no2", "o3"], ax=axes[0])
plot_seasonality(df, "no2", by="hour", ax=axes[1])
fig.tight_layout()
fig.savefig("overview.png", dpi=150, bbox_inches="tight")
```

```python
# Or let a single function own its figure and save directly:
plot_timeseries(df, save_path="timeseries.png")
```

Most functions return a single `Axes`. Two return a **NumPy array** of axes
because they own a grid: `plot_distributions` (one panel per column) and
`plot_residuals` (a two-panel diagnostic). `plot_correlation(cluster=True)`
returns a seaborn `ClusterGrid` instead of an `Axes`.

## EDA charts

These answer the first questions asked of a new dataset: what is missing, how
values are distributed, how variables correlate, how they evolve, and what
seasonal structure they carry. All accept an optional `columns=` argument
(default: all columns, or all *numeric* columns where a numeric type is
required) and never mutate the input frame.

### `plot_missingness`

```python
from nextaire_tools.viz import plot_missingness

plot_missingness(df)                     # bar of missing fraction per column
plot_missingness(df, kind="matrix")      # rows × columns map; missing cells dark
```

`kind="bar"` (default) sorts columns by missing fraction; `kind="matrix"` draws
a `SEQUENTIAL_CMAP` heatmap of the missing mask, which reveals *where* gaps sit
in time. Pair it with [Missing values](missing-values.md) to choose a fill
strategy.

### `plot_distributions`

```python
from nextaire_tools.viz import plot_distributions

axes = plot_distributions(df, bins=40, kde=True, ncols=3)
```

Draws a histogram (with optional KDE overlay) per numeric column in an
`ncols`-wide grid and returns the array of axes; unused cells are hidden.
Non-numeric columns are dropped automatically.

### `plot_correlation`

```python
from nextaire_tools.viz import plot_correlation

plot_correlation(df, method="pearson", annot=True)
grid = plot_correlation(df, method="spearman", cluster=True)   # returns a ClusterGrid
```

The heatmap uses `DIVERGING_CMAP` centered at zero (`vmin=-1`, `vmax=1`), so the
neutral gray midpoint means "no correlation". `method` is `"pearson"`,
`"spearman"`, or `"kendall"`. With `cluster=True` the matrix is hierarchically
reordered via `seaborn.clustermap` and a `ClusterGrid` is returned.

### `plot_timeseries`

```python
from nextaire_tools.viz import plot_timeseries

plot_timeseries(df, ["no2", "o3", "pm10"])
plot_timeseries(df, ["no2"], time_col="timestamp")   # if time is a column, not the index
```

Plots one line per numeric column against the datetime axis. With two or more
series a legend is drawn; a single series is named in the title. Pass
`time_col=` when timestamps live in a column rather than the index.

### `plot_seasonality`

```python
from nextaire_tools.viz import plot_seasonality

plot_seasonality(df, "o3", by="hour")        # diurnal cycle
plot_seasonality(df, "no2", by="dayofweek")  # weekly cycle
plot_seasonality(df, "pm10", by="month")     # annual cycle
```

Box-plots a single `column` grouped by a calendar component read from the
datetime index. `by` must be one of `"hour"`, `"dayofweek"`, or `"month"`.

!!! example "A five-minute EDA pass"
    ```python
    from nextaire_tools.viz import (
        plot_missingness, plot_distributions, plot_correlation,
        plot_timeseries, plot_seasonality,
    )

    plot_missingness(df, save_path="reports/missing.png")
    plot_distributions(df, save_path="reports/dists.png")
    plot_correlation(df, save_path="reports/corr.png")
    plot_timeseries(df, save_path="reports/series.png")
    plot_seasonality(df, "o3", by="hour", save_path="reports/o3_hour.png")
    ```

## Outlier charts

Two views that support the workflow in [Outliers](outliers.md): a multi-column
overview and a single-series close-up with the decision bounds drawn in.

### `plot_boxplots`

```python
from nextaire_tools.viz import plot_boxplots

plot_boxplots(df, ["no2", "o3", "pm10"])
```

Horizontal box-plots for the numeric columns — a fast scan for skew, spread,
and gross outliers before you commit to a detection method.

### `plot_outliers`

`plot_outliers(df, column, ...)` plots one series and highlights out-of-bounds
points in a reserved status red, with the bound lines as dashed horizontals.
Supply the bounds one of two ways:

=== "Explicit bounds"

    ```python
    from nextaire_tools.viz import plot_outliers

    plot_outliers(df, "no2", bounds=(0.0, 200.0))
    plot_outliers(df, "o3", bounds=(0.0, None))   # one-sided (lower only)
    ```

=== "A fitted handler"

    ```python
    from nextaire_tools.preprocessing import OutlierHandler
    from nextaire_tools.viz import plot_outliers

    handler = OutlierHandler(
        columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"
    ).fit(df)

    plot_outliers(df, "no2", handler=handler)     # reads handler.bounds_["no2"]
    ```

`bounds=` takes precedence over `handler=`. The `handler` is duck-typed: any
object exposing a fitted `bounds_` mapping keyed by column works, so `plot_outliers`
never imports from `nextaire_tools.preprocessing`. Either end of `bounds` may be `None`
for a one-sided limit.

!!! warning "Look before you drop"
    A spike may be a genuine pollution episode, not a sensor fault. `plot_outliers`
    is the tool for *inspecting* what a detector flags **before** you choose a
    strategy. Prefer `strategy="clip"` or `"flag"` over `"drop"` until you have
    eyeballed the flagged points. Note that `OutlierHandler(method="isolation_forest")`
    is multivariate (row-level) and does not produce per-column `bounds_`, so it
    cannot drive `plot_outliers` via `handler=` — pass explicit `bounds=` for a
    univariate view. See [Outliers](outliers.md) for the full decision guide.

## Model-evaluation charts

Diagnostics for a fitted regressor. Inputs are array-like (`y_true`, `y_pred`) —
NumPy arrays, pandas `Series`, or plain sequences — so these work with any
model. They pair naturally with [Modeling](modeling.md) and the metric
definitions in [Metrics glossary](../reference/metrics.md).

### `plot_predictions`

```python
from nextaire_tools.viz import plot_predictions

plot_predictions(y_test, y_pred, kind="scatter")            # predicted vs observed
plot_predictions(y_test, y_pred, kind="timeseries")         # both series overlaid
plot_predictions(y_test, y_pred, kind="timeseries", index=y_test.index)
```

`kind="scatter"` (default) plots predicted against observed with a dashed 1:1
reference line and the R² in the title. `kind="timeseries"` overlays the two
series; pass `index=` for the x-axis (or a `Series` `y_true` whose index is used
automatically).

### `plot_residuals`

```python
from nextaire_tools.viz import plot_residuals

axes = plot_residuals(y_test, y_pred)   # returns a length-2 array of axes
```

Two panels: residuals versus predicted (with a dashed zero line) and a histogram
of the residuals. Look for structure in panel 1 (fanning, curvature) and skew or
heavy tails in panel 2.

### `plot_feature_importance`

```python
from nextaire_tools.viz import plot_feature_importance

# From a fitted estimator (reads feature_importances_ or |coef_|):
plot_feature_importance(rf, top_n=15)

# Or from a raw array + names:
plot_feature_importance([0.5, 0.3, 0.2], ["no2_lag1", "hour_sin", "o3"])
```

Horizontal bar chart of the largest importances. Passed a fitted estimator, it
reads `feature_importances_` (trees) or `abs(coef_)` (linear models, averaged
across outputs when multi-output), and takes names from `feature_names_in_` when
available. Passed a raw array, supply `feature_names=` yourself. `top_n` caps how
many bars are shown.

!!! note "These charts don't fit models"
    `plot_feature_importance` reads an **already-fitted** estimator; it never
    calls `fit`. Build and validate the model with the
    [modeling workflow](modeling.md) first, then hand the fitted object here.

## See also

- [Outliers](outliers.md) — detection methods and the `OutlierHandler` bounds
  that feed `plot_outliers`.
- [Missing values](missing-values.md) — strategies to act on what
  `plot_missingness` reveals.
- [Modeling & cross-validation](modeling.md) — produces the `y_true`/`y_pred`
  that the evaluation charts consume.
- [`nextaire_tools.viz` API reference](../api/viz.md) — full signatures and return types.
