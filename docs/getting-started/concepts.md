# Key concepts

`nextaire_tools` is built on one small, consistent idea: **everything that touches your
data is a *step*, and steps compose into a *pipeline* that flows a single
DataFrame from raw table to model-ready feature matrix.** This page explains that
mental model, why steps are DataFrame-in / DataFrame-out, the one place where a
`nextaire_tools.Pipeline` deliberately differs from scikit-learn's, and how optional
dependencies are gated. Once these click, the rest of the library is predictable.

## Everything is a step

A *step* is a scikit-learn-compatible transformer that subclasses
[`BaseStep`](../api/preprocessing.md). Every step —
`MissingValueHandler`, `OutlierHandler`, `TemporalFeatures`, `Scaler` — shares
the same three-method contract:

```python
step.fit(X)            # learn state from X; returns self
step.transform(X)      # apply the learned transformation; returns a new DataFrame
step.fit_transform(X)  # fit then transform in one call
```

Three rules hold for **every** step, so you never have to memorize per-class
behavior:

1. **DataFrame in, DataFrame out.** `transform` takes a `pandas.DataFrame` and
   returns a new one.
2. **The input is never mutated.** Steps work on a defensive copy, so your
   original frame is safe.
3. **Learned state lives in trailing-underscore attributes.** Anything a step
   discovers during `fit` is stored on `self` with a trailing `_`, following the
   scikit-learn convention. Constructor arguments have no underscore; fitted
   state does.

```python
import pandas as pd
from nextaire_tools.preprocessing import MissingValueHandler

df = pd.DataFrame({"no2": [1.0, None, 3.0], "o3": [None, 2.0, 3.0]})

mvh = MissingValueHandler(strategy="mean")   # a *parameter*, no underscore
out = mvh.fit_transform(df)

mvh.statistics_        # learned fill values      -> {'no2': 2.0, 'o3': 2.5}
mvh.missing_fraction_  # per-column NaN fraction  (a pandas Series)
mvh.feature_names_in_  # columns seen during fit  -> array(['no2', 'o3'], ...)
```

Every fitted step also exposes `feature_names_in_`, `n_features_in_`, and
`columns_` (the resolved list of columns it operates on). Calling `transform`
before `fit` raises `NotFittedError`. The state each step learns is documented on
its API page — for example `bounds_` / `outlier_fraction_` on
[`OutlierHandler`](../api/preprocessing.md), `scaler_` on `Scaler`, and
`feature_names_out_` on `TemporalFeatures`.

!!! note "Steps that add or remove columns"
    Most steps keep the columns unchanged, but some reshape the frame:
    `TemporalFeatures` **adds** columns, `MissingValueHandler` can **drop**
    columns (via `column_missing_threshold`) or add `__missing` indicators, and
    both `OutlierHandler` and `MissingValueHandler` can **drop rows**. Each such
    step overrides `get_feature_names_out()` so the output schema is always
    introspectable.

## Steps compose into a Pipeline

A [`Pipeline`](../api/preprocessing.md) is an ordered list of steps. Its `fit` /
`transform` / `fit_transform` thread one DataFrame through the steps in
sequence — the output of each step is the input to the next:

```python
from nextaire_tools import Pipeline
from nextaire_tools.preprocessing import MissingValueHandler, Scaler

pipe = Pipeline([
    MissingValueHandler(strategy="mean"),
    Scaler(method="standard"),
])
clean = pipe.fit_transform(df)
```

Because a `Pipeline` *is* just a sequence of steps, it is easy to introspect and
slice:

```python
pipe.named_steps           # {'missingvaluehandler0': ..., 'scaler1': ...}
pipe.steps_                # the normalized [(name, step), ...] list
pipe[0]                    # first step, by position
pipe["scaler1"]            # a step, by name
pipe[:1]                   # a *new* Pipeline with just the first step
len(pipe)                  # 2
```

You can build a pipeline three ways: pass steps directly (auto-named, or as
`(name, step)` tuples), use `make_pipeline(*steps)`, or construct one
declaratively with `Pipeline.from_config([...])`, which resolves class names
through the `STEP_REGISTRY`. See [Pipelines](../user-guide/pipelines.md) for all
three.

## The DataFrame flows through — keep the target as a column

The unit that flows through a `nextaire_tools` pipeline is **one DataFrame**, not the
`(X, y)` pair that scikit-learn passes around. The practical convention that
follows is: **keep your target (e.g. `no2`) as a column of the frame** while you
clean and engineer features, and only split it out just before modeling:

```python
clean = pipe.fit_transform(df)      # target is still a column here
X, y = clean.drop(columns="no2"), clean["no2"]   # split at the very end
```

### Why DataFrame-in / DataFrame-out?

A bare scikit-learn transformer returns a NumPy array, which **discards column
names, dtypes, and the index**. For air-quality work that metadata is not
incidental — it is load-bearing:

- **Column names** let steps target specific pollutants (`columns=["no2", "o3"]`)
  and let you read a feature matrix months later.
- **The `DatetimeIndex`** is what makes `MissingValueHandler`'s `"interpolate"`
  time-aware and lets `TemporalFeatures` derive `hour` / `season` / cyclical
  encodings straight from the timestamp.

Keeping a DataFrame end-to-end means these survive every step, and the frame you
hand to a model is self-describing.

!!! danger "Checkpoint — target alignment with row-dropping steps"
    Some steps change the **number of rows**: `MissingValueHandler(strategy="drop")`
    and `OutlierHandler(strategy="drop")`. Inside a `nextaire_tools.Pipeline` this is
    safe, because the target rides along as a column and stays row-aligned with
    its features.

    It is **not** safe inside `sklearn.pipeline.Pipeline` when you pass a
    separate `y`: the transformer drops rows from `X` but scikit-learn does not
    drop the matching entries from `y`, so `X` and `y` silently fall out of
    alignment and every downstream label is wrong.

    ```python
    # OK — nextaire_tools flows one frame; dropped rows take their target with them
    Pipeline([MissingValueHandler(strategy="drop")]).fit_transform(clean)

    # DANGER — X loses rows, y does not; they no longer line up
    # sklearn.pipeline.Pipeline([...]).fit(X, y)
    ```

    Rule of thumb: do your **row-dropping** inside a `nextaire_tools.Pipeline` (or as an
    explicit `df.dropna()` on the whole frame), then split `X` / `y`.

## Optional dependencies and `require()`

`nextaire_tools` keeps `import nextaire_tools` fast and dependency-light. Heavy subpackages
(`nextaire_tools.viz`, `nextaire_tools.models`, `nextaire_tools.extractors`) are imported lazily on first
access, and features that need an optional package do **not** fail at import
time. Instead, the moment you call into such a feature, a small internal
`require()` helper tries to import the dependency and, if it is missing, raises a
`MissingDependencyError` that names the exact extra to install:

```text
The optional dependency 'torch' is required for deep-learning models but is not
installed.
Install it with:  pip install 'nextaire_tools[deep]'
```

This gating means you can:

- `import nextaire_tools`, `import nextaire_tools.models`, and `import nextaire_tools.extractors` with only
  the core install;
- write code that references `LSTMRegressor` or `CopernicusExtractor` without
  those extras present;

and only hit the requirement when you actually **fit** the deep model, **call**
an extractor, or request the `is_holiday` temporal feature (which needs the
`holidays` extra). `MissingDependencyError` is a subclass of both `NextaireToolsError`
and the built-in `ImportError`, so you can catch it either way. Run `nextaire_tools info`
to see which extras are present in your environment — see
[Installation](installation.md#verifying-the-install) for the extras table and
the `require()`-gated feature list.

## Putting it together

- A **step** learns state in `fit` (stored in `trailing_underscore_` attributes)
  and returns a new DataFrame in `transform`.
- A **`Pipeline`** chains steps, flowing one DataFrame through them.
- The **target stays a column** so row-dropping steps keep features and labels
  aligned — split `X` / `y` only at the end.
- **DataFrame-in / DataFrame-out** preserves the column names and
  `DatetimeIndex` the whole pipeline (and your models) rely on.
- **Optional dependencies** are gated by `require()`, so the core stays light and
  errors are actionable.

Ready to apply it? Head to the [Quickstart](quickstart.md) for an end-to-end run,
or the [user guide](../user-guide/pipelines.md) for one page per step.
