# Pipelines

[`Pipeline`](../api/preprocessing.md) chains preprocessing steps into a single
`DataFrame`-in / `DataFrame`-out unit. Unlike
[`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
it threads **one frame** through every step (never a separate `X`/`y` pair),
which keeps row-dropping steps — outlier drops, missing-value drops — perfectly
aligned with the rest of the data.

```python
from nextaire_tools import Pipeline
from nextaire_tools.preprocessing import (
    MissingValueHandler, OutlierHandler, TemporalFeatures, Scaler, make_pipeline,
)
```

We reuse the hourly `no2`/`o3`/`pm10` frame from [Loading data](loading-data.md).

## Building a pipeline

Pass an ordered list of steps. Each element is either a bare step (auto-named)
or a `(name, step)` tuple:

```python
pipe = Pipeline([
    ("impute",        MissingValueHandler(strategy="interpolate", limit=3)),
    ("drop_residual", MissingValueHandler(strategy="drop")),
    ("outliers",      OutlierHandler(columns=["no2", "o3", "pm10"],
                                     method="iqr", strategy="clip")),
    ("time",          TemporalFeatures(cyclical=("hour", "dayofweek", "dayofyear"))),
    ("scale",         Scaler(columns=["no2", "o3", "pm10"], method="standard")),
])
clean = pipe.fit_transform(df)
```

`fit`, `transform`, and `fit_transform` all thread the transformed frame from one
step to the next; `fit_transform(df)` equals `fit(df).transform(df)`.

Bare steps are auto-named as the lowercased class name plus its position index —
so [`make_pipeline`](../api/preprocessing.md), which auto-names for you, produces
`missingvaluehandler0`, `scaler1`:

```python
pipe = make_pipeline(MissingValueHandler(strategy="mean"), Scaler())
list(pipe.named_steps)   # ['missingvaluehandler0', 'scaler1']
```

!!! note "Construction is validated"
    A `ConfigurationError` is raised if `steps` is empty, contains a
    non-`BaseStep` object, or has duplicate step names.

## Accessing steps

`named_steps` maps names to instances, and a pipeline supports integer, string,
and slice indexing. Slicing returns a **new** `Pipeline`:

```python
pipe.named_steps["outliers"]   # the OutlierHandler instance
pipe[0]                        # first step (by position)
pipe["scale"]                  # a step (by name)
pipe[:2]                       # a new Pipeline of the first two steps
len(pipe)                      # number of steps
```

This is handy for inspecting fitted state after `fit`:

```python
pipe.fit(df)
pipe["outliers"].bounds_          # learned per-column bounds
pipe["impute"].missing_fraction_  # gap fractions
```

## Declarative pipelines with `from_config`

`Pipeline.from_config` builds a pipeline from a list of plain dicts, resolving
step names through the module-level `STEP_REGISTRY`
(`MissingValueHandler`, `OutlierHandler`, `TemporalFeatures`, `Scaler`). Each
entry needs a `"step"` key and may carry `"params"` and an optional `"name"`:

```python
config = [
    {"step": "MissingValueHandler", "params": {"strategy": "interpolate", "limit": 3}},
    {"step": "OutlierHandler", "params": {"method": "iqr", "strategy": "clip"}},
    {"step": "TemporalFeatures", "params": {"cyclical": ["hour", "dayofyear"]}},
    {"step": "Scaler", "params": {"method": "standard"}},
]
pipe = Pipeline.from_config(config)
clean = pipe.fit_transform(df)
```

Because the config is pure data, it serializes to JSON — which is exactly what
the CLI consumes.

### From the command line

The `nextaire_tools preprocess` command runs a pipeline over a file and writes the
result. Point `--config` at a JSON file whose top-level object has a `"steps"`
list (a bare list also works):

```json title="clean.json"
{
  "steps": [
    {"step": "MissingValueHandler", "params": {"strategy": "interpolate", "limit": 3}},
    {"step": "OutlierHandler", "params": {"method": "iqr", "strategy": "clip"}},
    {"step": "TemporalFeatures"},
    {"step": "Scaler", "params": {"method": "standard"}}
  ]
}
```

```bash
nextaire_tools preprocess station.csv clean.parquet \
    --config clean.json --time-col timestamp --set-time-index
```

Without `--config`, `preprocess` runs a sensible default pipeline (bounded
interpolation → IQR clip → temporal features). An unknown step name or a
malformed entry raises `ConfigurationError`.

## Output feature names

`get_feature_names_out()` delegates to the final step, so it reports the schema
of the frame the pipeline produces:

```python
pipe.fit(df)
list(pipe.get_feature_names_out())
# ['no2', 'o3', 'pm10', 'hour', 'hour_sin', 'hour_cos', ...]
```

(It raises `AttributeError` only if the last step lacks the method; every built-in
nextaire_tools step implements it.)

## Keeping features and target aligned

!!! danger "Checkpoint — keep the target as a column"
    Steps like `OutlierHandler(strategy="drop")` and
    `MissingValueHandler(strategy="drop")` change the **number of rows**. If your
    features `X` and target `y` are separate objects, a dropped row falls out of
    `X` but not `y`, and the two silently misalign — every subsequent metric is
    computed against shifted labels.

    The nextaire_tools convention avoids this: **keep the target as a column of the
    frame** and run the whole pipeline on that one frame, so row drops apply to
    features and target together. Split into `X`/`y` *after* preprocessing:

    ```python
    clean = pipe.fit_transform(df)          # target still a column, rows aligned
    y = clean["no2"]
    X = clean.drop(columns="no2")
    ```

    For the same reason, do **not** put a row-dropping nextaire_tools step inside an
    `sklearn.pipeline.Pipeline` that carries a separate `y` — sklearn will not
    drop the matching labels. Use `nextaire_tools.Pipeline` for anything that changes the
    row count.

!!! warning "Fit the pipeline on training rows only"
    A preprocessing pipeline learns state (imputation values, scaler statistics,
    outlier bounds). Fit it on the **training** slice and only `transform` the
    test slice, or test information leaks into training. Split time series
    chronologically — never shuffle — with the tools in the
    [modeling guide](modeling.md); `sklearn.model_selection.train_test_split(shuffle=True)`
    silently inflates scores on temporal data.

!!! example "A complete clean → feature → scale pipeline"
    ```python
    from nextaire_tools import load_table, Pipeline
    from nextaire_tools.preprocessing import (
        MissingValueHandler, OutlierHandler, TemporalFeatures, Scaler,
    )

    df = load_table("station.csv", time_col="timestamp", set_time_index=True)

    pipe = Pipeline([
        ("impute",  MissingValueHandler(strategy="interpolate", limit=3)),
        ("drop",    MissingValueHandler(strategy="drop")),          # clear residual NaNs
        ("outlier", OutlierHandler(columns=["no2", "o3", "pm10"],
                                   method="iqr", strategy="clip")),
        ("time",    TemporalFeatures(cyclical=("hour", "dayofweek", "dayofyear"))),
        ("scale",   Scaler(columns=["no2", "o3", "pm10"], method="standard")),
    ])

    clean = pipe.fit_transform(df)
    assert clean[["no2", "o3", "pm10"]].isna().to_numpy().sum() == 0
    ```

## See also

- [Missing values](missing-values.md), [Outliers](outliers.md),
  [Temporal features](temporal-features.md), [Scaling](scaling.md) — the steps a
  pipeline chains.
- [Modeling & cross-validation](modeling.md) — leakage-free time-series splits.
- [API reference: `Pipeline`, `make_pipeline`, `STEP_REGISTRY`](../api/preprocessing.md).
