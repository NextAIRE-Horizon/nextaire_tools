# Metrics glossary

Every score returned by
[`regression_metrics`](../api/models.md) — with its exact formula, range, units,
how to read it, and when it will quietly mislead you. Two of them
(**index of agreement** and **FAC2**) are the atmospheric-science standard for
air-quality model evaluation and deserve a place in every report.

Throughout, \(o_i\) is the observed value, \(p_i\) the prediction, \(n\) the
number of valid pairs, and \(\bar o = \tfrac{1}{n}\sum_i o_i\) the mean of the
observations. `regression_metrics` **drops any pair where either value is
`NaN`** before scoring, so gappy observation series are handled transparently; a
metric that is undefined for the given data (e.g. correlation on a single point)
is returned as `float("nan")`.

---

## Error magnitude

### `mae` — mean absolute error

\[
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert p_i - o_i \rvert
\]

- **Range / units:** \([0, \infty)\), in the target's own units. `0` is perfect.
- **Interpretation:** the typical absolute miss. Easy to explain ("on average
  we are off by X µg/m³") and robust — every error contributes linearly.
- **Misleading when:** you care specifically about large excursions. MAE weights
  a 100-unit miss the same per unit as a 1-unit miss, so it can look fine while
  the model blows the rare high-pollution peaks.

### `mse` — mean squared error

\[
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n} (p_i - o_i)^2
\]

- **Range / units:** \([0, \infty)\), in the target's units **squared**.
- **Interpretation:** mainly an optimization target and the building block of
  RMSE. Large errors dominate because they are squared.
- **Misleading when:** you report it directly — the squared units are not
  intuitive. Prefer RMSE for communication.

### `rmse` — root mean squared error

\[
\mathrm{RMSE} = \sqrt{\mathrm{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (p_i - o_i)^2}
\]

- **Range / units:** \([0, \infty)\), back in the target's units. `0` is perfect.
- **Interpretation:** like MAE but penalizes big misses more; always
  \(\mathrm{RMSE} \ge \mathrm{MAE}\), and the gap between them grows with error
  variance.
- **Misleading when:** the series has heavy tails or a few genuine spikes — a
  handful of large residuals can dominate RMSE and hide otherwise good typical
  performance. Report it *alongside* MAE, not instead of it.

---

## Skill relative to a baseline

### `r2` — coefficient of determination

\[
R^2 = 1 - \frac{\sum_{i}(p_i - o_i)^2}{\sum_{i}(o_i - \bar o)^2}
\]

- **Range / units:** \((-\infty, 1]\), dimensionless. `1` is perfect, `0` means
  "no better than always predicting \(\bar o\)", negative means "worse than the
  mean". Returned as `NaN` when the observations have zero variance.
- **Interpretation:** the fraction of observed variance the model explains,
  measured against the mean baseline. Unlike Pearson \(r\), it *does* penalize
  bias (the numerator is the raw squared error).
- **Misleading when:** the series is highly variable. A strongly seasonal
  pollutant has a large denominator, so even a mediocre model can post a
  flattering \(R^2\) simply by tracking the obvious daily swing. Judge \(R^2\)
  against how easy the mean baseline is to beat.

---

## Percentage error

### `mape` — mean absolute percentage error

\[
\mathrm{MAPE} = \frac{100}{n_{\neq 0}} \sum_{\,o_i \neq 0} \left\lvert \frac{p_i - o_i}{o_i} \right\rvert
\]

where the sum runs only over observations that are non-zero (\(n_{\neq 0}\) of
them); it is `NaN` if every observation is zero.

- **Range / units:** \([0, \infty)\), in percent.
- **Interpretation:** average relative error, convenient for comparing across
  pollutants on different scales.
- **Misleading when:** observations are **near zero** — a clean-air hour with
  \(o_i \approx 0\) produces an enormous percentage error that swamps the mean.
  MAPE is also **asymmetric**: it penalizes over-prediction more than
  under-prediction, and it silently ignores all zero observations. For
  air-quality data, which routinely dips toward zero at night, treat MAPE with
  suspicion.

### `smape` — symmetric mean absolute percentage error

\[
\mathrm{sMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{2\,\lvert p_i - o_i \rvert}{\lvert p_i \rvert + \lvert o_i \rvert}
\]

Terms whose denominator is zero contribute `0`.

- **Range / units:** \([0, 200]\) percent (this two-sided form is bounded).
- **Interpretation:** a bounded, more balanced relative error than MAPE;
  degrades gracefully rather than exploding as observations shrink.
- **Misleading when:** *both* \(p_i\) and \(o_i\) are close to zero, where the
  ratio is still unstable. Despite the name it is not perfectly symmetric, so do
  not over-interpret small differences.

---

## Direction of error

### `bias` — mean signed error

\[
\mathrm{bias} = \frac{1}{n}\sum_{i=1}^{n} (p_i - o_i)
\]

- **Range / units:** \((-\infty, \infty)\), in the target's units.
- **Interpretation:** systematic offset. **Positive means the model
  over-predicts** on average; negative means it under-predicts. Essential for
  regulatory work, where a consistent under-prediction of exceedances is a
  safety issue.
- **Misleading when:** read alone. Positive and negative errors cancel, so a
  bias near zero says nothing about accuracy — a wildly scattered model can be
  perfectly unbiased. Always pair it with MAE/RMSE.

---

## Association

### `pearson_r` — linear correlation

\[
r = \frac{\sum_i (o_i - \bar o)(p_i - \bar p)}{\sqrt{\sum_i (o_i - \bar o)^2}\;\sqrt{\sum_i (p_i - \bar p)^2}}
\]

- **Range / units:** \([-1, 1]\), dimensionless. `1` is perfect positive linear
  association. Returned as `NaN` when \(n < 2\) or either series is constant.
- **Interpretation:** how well the prediction tracks the *shape* of the
  observations.
- **Misleading when:** used as an accuracy score. Pearson \(r\) is invariant to
  scale and offset — a prediction that is exactly twice the truth, or shifted by
  a constant, still scores \(r = 1\). It ignores bias entirely. This is precisely
  why atmospheric evaluation prefers the index of agreement below.

### `spearman_r` — rank correlation

\[
\rho = r\big(\operatorname{rank}(o),\, \operatorname{rank}(p)\big)
\]

i.e. Pearson correlation computed on the ranks (ties handled by
`scipy.stats.spearmanr`).

- **Range / units:** \([-1, 1]\), dimensionless. `NaN` when \(n < 2\).
- **Interpretation:** captures any **monotonic** relationship, not just linear;
  robust to outliers and nonlinear-but-order-preserving distortions.
- **Misleading when:** you need magnitude fidelity — \(\rho\) sees only ordering,
  so a model that ranks hours correctly but gets every value badly wrong still
  scores highly.

---

## Air-quality standard metrics

!!! tip "Report these two"
    In atmospheric-science model evaluation, the **index of agreement** and
    **FAC2** are reported as a matter of course, usually alongside RMSE and bias.
    Include them whenever you present an air-quality model — reviewers expect
    them, and they expose failure modes (bias, factor-scale error) that R² and
    Pearson \(r\) hide.

### `index_of_agreement` — Willmott's *d*

\[
d = 1 - \frac{\displaystyle\sum_{i}(p_i - o_i)^2}{\displaystyle\sum_{i}\big(\lvert p_i - \bar o \rvert + \lvert o_i - \bar o \rvert\big)^2}
\]

- **Range / units:** \([0, 1]\), dimensionless. `1` is perfect agreement; `0` is
  complete disagreement. `NaN` when the denominator is zero.
- **Interpretation:** a standardized measure of model error, normalized by the
  *largest* error the prediction and observation could plausibly have about the
  observed mean. Unlike Pearson \(r\), it responds to both additive bias and
  proportional (scale) differences, which is why Willmott introduced it for
  environmental models.
- **Misleading when:** the denominator depends on observed variability, so — like
  \(R^2\) — a very high-variance series can inflate *d*. Read it together with
  `bias` to catch a systematic offset that a high *d* might otherwise mask.

### `fac2` — factor-of-two fraction

\[
\mathrm{FAC2} = \frac{1}{n_{\neq 0}}\;\#\left\{\, i \;:\; 0.5 \le \frac{p_i}{o_i} \le 2 \,\right\}
\]

over the non-zero observations (zeros are ignored; `NaN` if all observations are
zero).

- **Range / units:** \([0, 1]\), dimensionless — a fraction of points.
- **Interpretation:** the share of predictions within a factor of two of the
  observation. It answers the blunt operational question "how often are we in the
  right ballpark?" A common acceptance benchmark for dispersion models is
  \(\mathrm{FAC2} \ge 0.5\).
- **Misleading when:** used as your only score. FAC2 is **coarse** — a ratio of
  1.99 counts exactly like 1.0 — so a systematically biased model can still post
  a high FAC2 as long as it stays inside the 2× band. It also discards zero
  observations entirely.

---

## Using `regression_metrics`

Pass observed and predicted array-likes (NumPy arrays, `pandas.Series`, or plain
sequences). Inputs are flattened and paired element-wise; unequal lengths raise
`SchemaError`.

```python
import numpy as np
from nextaire_tools.models import regression_metrics

obs  = np.array([12.0, 18.0, 25.0,  9.0, 30.0, 22.0])
pred = np.array([14.0, 16.0, 24.0, 11.0, 27.0, 20.0])

regression_metrics(obs, pred)
```

```python
{'mae': 2.0,
 'mse': 4.333333333333333,
 'rmse': 2.0816659994661326,
 'r2': 0.9174950298210736,
 'mape': 12.181818181818182,
 'smape': 11.880234249710884,
 'bias': -0.6666666666666666,
 'pearson_r': 0.9860132971832694,
 'spearman_r': 1.0,
 'index_of_agreement': 0.9735894495114006,
 'fac2': 1.0}
```

The negative `bias` shows a slight average **under**-prediction; `spearman_r == 1`
confirms the ranking is perfect even though a few values are off; and both `fac2`
and the index of agreement are near their ideal `1`.

Request a subset with `metrics=[...]` (unknown names raise `ConfigurationError`);
the canonical full ordering lives in `nextaire_tools.models.evaluate.METRIC_NAMES`:

```python
regression_metrics(obs, pred, metrics=["rmse", "index_of_agreement", "fac2"])
# {'rmse': 2.0816659994661326, 'index_of_agreement': 0.9735894495114006, 'fac2': 1.0}
```

The same subset flows into per-fold cross-validation via
[`cross_val_report`](../api/models.md), which tabulates each metric per fold plus
`mean` and `std` rows:

```python
from nextaire_tools.models import cross_val_report, make_regressor, BlockingTimeSeriesSplit

report = cross_val_report(
    make_regressor("random_forest"), X, y,
    cv=BlockingTimeSeriesSplit(n_splits=4, gap=1),
    metrics=["mae", "rmse", "index_of_agreement", "fac2"],
)
```

---

## Quick reference

| Metric | Formula range | Best | Units | Watch out |
| --- | --- | --- | --- | --- |
| `mae` | \([0,\infty)\) | `0` | target | under-weights rare large errors |
| `mse` | \([0,\infty)\) | `0` | target² | unintuitive units |
| `rmse` | \([0,\infty)\) | `0` | target | dominated by a few big misses |
| `r2` | \((-\infty,1]\) | `1` | — | flattered by high-variance series |
| `mape` | \([0,\infty)\) % | `0` | % | explodes near zero; asymmetric |
| `smape` | \([0,200]\) % | `0` | % | unstable when both ≈ 0 |
| `bias` | \((-\infty,\infty)\) | `0` | target | errors cancel — never read alone |
| `pearson_r` | \([-1,1]\) | `1` | — | blind to bias and scale |
| `spearman_r` | \([-1,1]\) | `1` | — | ignores magnitude |
| `index_of_agreement` | \([0,1]\) | `1` | — | pair with `bias` |
| `fac2` | \([0,1]\) | `1` | — | coarse; ignores zero obs |

See the [modeling guide](../user-guide/modeling.md) for choosing a splitter and the
[end-to-end tutorial](../tutorials/end-to-end-workflow.md) for these metrics in
context.
