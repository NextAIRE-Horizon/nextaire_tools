# Deep learning

`nextaire_tools` ships three PyTorch regressors — a multilayer perceptron
(`MLPRegressor`), an LSTM (`LSTMRegressor`), and a 1-D CNN (`CNNRegressor`) —
behind the same scikit-learn-style `fit`/`predict` API as the classical models
in [Modeling](modeling.md). Because they follow that contract, they interchange
with `make_regressor` output and flow straight through `cross_val_report`. A
Prophet wrapper (`ProphetForecaster`) is also available for pure time-series
forecasting.

See the [`nextaire_tools.models` API reference](../api/models.md) for full signatures.

!!! note "PyTorch is an optional extra"
    The neural models require the `deep` extra; Prophet requires `forecast`:

    ```bash
    pip install "nextaire_tools[deep]"        # PyTorch (MLP / LSTM / CNN)
    pip install "nextaire_tools[forecast]"    # Prophet
    ```

    Both dependencies are imported **lazily**, only when you actually fit a
    model — so `import nextaire_tools.models` succeeds with just the core install, and
    `make_sequences` works without PyTorch. If the extra is missing, fitting
    raises `MissingDependencyError` with the install hint.

These three architectures mirror those in the author's AAQR (2024) study on
hourly pollutant forecasting (LSTM / CNN / MLP); see [Citing nextaire_tools](../about/citation.md).

## Windowing with `make_sequences`

Sequence models look back over a window of recent hours. `make_sequences` turns
a 2-D feature matrix into the 3-D tensor those models consume:

```python
import numpy as np
from nextaire_tools.models import make_sequences

X = np.arange(10).reshape(-1, 1).astype(float)   # 10 timesteps, 1 feature
X3d, y2d = make_sequences(X, X.ravel(), window=3, horizon=1)
X3d.shape, y2d.shape
# ((7, 3, 1), (7, 1))
y2d[0, 0]        # target = value one step after the first window (rows 0,1,2)
# 3.0
```

`make_sequences(X, y=None, *, window, horizon=1)`:

- Each window covers `window` consecutive rows, `X[k : k + window]`.
- When `y` is given, the target for window `k` is the value `horizon` steps
  after the window's last row, i.e. `y[k + window - 1 + horizon]`. `horizon=1`
  predicts the next step.
- The number of windows is `n_samples - window - horizon + 1`, **the same
  whether or not `y` is supplied** — so windows built for prediction align
  exactly with those built for training.
- Shapes: `X3d` is `(n_windows, window, n_features)`; `y2d` is `(n_windows, 1)`
  (or `None` when `y` is omitted).

You rarely call this directly — `LSTMRegressor` and `CNNRegressor` window their
input internally — but it is the contract those models use, and it is useful for
building your own tensors.

## `MLPRegressor` vs `LSTMRegressor` / `CNNRegressor`

The essential difference is the input shape each expects and what `predict`
returns.

| Model | Input to `fit` | Windowed? | `predict` output |
| --- | --- | --- | --- |
| `MLPRegressor` | 2-D `(n_samples, n_features)` | No | length `n_samples`, all rows valid |
| `LSTMRegressor` | 2-D `(n_samples, n_features)` | Yes (internal) | length `n_samples`, **first `window` rows NaN** |
| `CNNRegressor` | 2-D `(n_samples, n_features)` | Yes (internal) | length `n_samples`, **first `window` rows NaN** |

All three take a plain 2-D feature matrix at `fit` — you never hand them a 3-D
tensor. The windowed models build sequences internally, and their `predict`
returns **one value per input row** with the first `window` rows set to `NaN`
(those rows lack a full look-back window). This keeps `len(pred) == len(X)` so
predictions line up positionally with your frame's index.

```python
import numpy as np
from nextaire_tools.models import MLPRegressor, LSTMRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.1, 500)

# Dense MLP on 2-D features.
mlp = MLPRegressor(hidden_sizes=(64, 32), epochs=100, random_state=0).fit(X, y)
mlp.predict(X).shape            # (500,) — every row valid

# Windowed LSTM: same 2-D X in, but the first `window` predictions are NaN.
lstm = LSTMRegressor(window=24, hidden_size=64, epochs=100, random_state=0).fit(X, y)
pred = lstm.predict(X)
pred.shape                      # (500,)
np.isnan(pred[:24]).all()       # True
```

`CNNRegressor` shares the LSTM's windowing and NaN-padding contract; it treats
features as input channels and the window as the temporal axis, with
`channels=(32, 32)` and `kernel_size=3` by default.

### Constructors at a glance

```python
MLPRegressor(*, hidden_sizes=(64, 32), dropout=0.1, epochs=100, batch_size=32,
             lr=1e-3, validation_fraction=0.1, patience=10, random_state=0)

LSTMRegressor(*, window=24, hidden_size=64, num_layers=1, dropout=0.1,
              epochs=100, batch_size=32, lr=1e-3, validation_fraction=0.1,
              patience=10, random_state=0)

CNNRegressor(*, window=24, channels=(32, 32), kernel_size=3, dropout=0.1,
             epochs=100, batch_size=32, lr=1e-3, validation_fraction=0.1,
             patience=10, random_state=0)
```

## The training contract

All three share the same machinery, which is worth understanding because it
changes how you preprocess:

- **Internal standardization.** At `fit`, each model stores the per-feature mean
  and standard deviation of `X` (and of `y`) and standardizes internally;
  `predict` un-scales back to the original units. You do **not** need to put a
  `Scaler` on the features feeding these models (see the checkpoint below).
- **Early stopping.** Training uses Adam + MSE loss and holds out a
  *chronological tail* of size `validation_fraction` for early stopping; if the
  validation loss does not improve for `patience` epochs, training stops and the
  best weights are restored. Set `validation_fraction=0` to disable it and train
  for the full `epochs`.
- **Determinism.** Given `random_state`, weight initialization and mini-batch
  ordering are deterministic on CPU, so repeated fits reproduce. (GPU kernels may
  introduce small nondeterminism.)
- **Fitted attributes.** After `fit`: `module_` (the trained `torch.nn.Module`)
  and `n_features_in_`.

## Cross-validating deep models

Because they expose scikit-learn `get_params`/`set_params` (via `BaseEstimator`),
these models clone cleanly and flow through `cross_val_report` with the same
leakage-free splitters as any other regressor:

```python
from nextaire_tools.models import LSTMRegressor, cross_val_report, ExpandingWindowSplit

report = cross_val_report(
    LSTMRegressor(window=24, epochs=50, random_state=0),
    X, y,
    cv=ExpandingWindowSplit(initial_train_size=24 * 60, test_size=24 * 7),
    metrics=["rmse", "r2", "index_of_agreement"],
)
```

!!! tip "Windowed models and per-fold NaNs"
    On each fold, a windowed model's `predict` returns `NaN` for the first
    `window` rows of that test block. This is not a problem for scoring:
    `regression_metrics` (and therefore `cross_val_report`) drops NaN pairs
    before computing anything. Just make sure each **test window is larger than
    `window`** so some valid predictions remain — otherwise the fold has nothing
    to score. Prefer `ExpandingWindowSplit`/`SlidingWindowSplit` with a
    `test_size` comfortably above `window`.

## Prophet forecasting

`ProphetForecaster` wraps Facebook Prophet for additive trend + seasonality +
holiday forecasting, which suits the strong daily/weekly cycles in urban
air-quality data. It accepts a time-indexed `Series`, a `ds`/`y` DataFrame, or
explicit `ds=`/`y=` arrays, and `predict` returns a frame with the forecast and
its uncertainty interval.

```python
from nextaire_tools.models import ProphetForecaster

# df is datetime-indexed; a single pollutant Series is the simplest input.
fc = ProphetForecaster(daily_seasonality=True, weekly_seasonality=True).fit(df["no2"])

forecast = fc.predict(periods=24, freq="H")     # 24 hours ahead
list(forecast.columns)
# ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
```

`ProphetForecaster(*, growth="linear", yearly_seasonality="auto",
weekly_seasonality="auto", daily_seasonality="auto", **kwargs)`. Extra keyword
arguments (e.g. `changepoint_prior_scale`, `interval_width`) pass straight
through to the underlying `Prophet` constructor. In `predict`, omit `periods`
(and `future`) to score the training timestamps only, or pass an explicit
`future=` frame/index to predict on arbitrary dates.

!!! danger "Checkpoint: fix NaNs before a deep model — but don't double-scale"
    Two failure modes bite here:

    1. **NaNs propagate.** These models standardize and window their input with
       NumPy/PyTorch; a single `NaN` in `X` or `y` contaminates the standard
       deviation and the loss. Run a `MissingValueHandler` to remove NaNs
       **before** fitting (interpolate short gaps, then drop the rest — see
       [Missing values](missing-values.md)). This is the same "interpolate then
       drop" trap called out in [Modeling](modeling.md).
    2. **Redundant scaling.** The models already standardize internally, so a
       feature `Scaler` in front of them is unnecessary — it will simply be
       re-standardized and buys nothing. (It is harmless, not required.) You
       *do* still want to clean and, where relevant, encode cyclical time with
       [`TemporalFeatures`](temporal-features.md) before windowing.

!!! warning "Small-data caveat"
    Neural networks need enough data to earn their capacity. On a few weeks of
    hourly data, a `random_forest` or `hist_gradient_boosting` from
    [`make_regressor`](modeling.md) will usually match or beat these models with
    far less tuning and full reproducibility. Reach for MLP/LSTM/CNN when you have
    long records (many months to years) and a clear temporal structure the
    windowed models can exploit — and always validate against a classical
    baseline with `cross_val_report`.

## See also

- [Modeling & cross-validation](modeling.md) — splitters, metrics,
  `cross_val_report`, and the classical baselines to compare against.
- [Missing values](missing-values.md) and
  [Temporal features](temporal-features.md) — the preprocessing these models
  depend on.
- [Visualization](visualization.md) — `plot_predictions` / `plot_residuals` for
  the predicted vs observed diagnostics.
- [`nextaire_tools.models` API reference](../api/models.md).
