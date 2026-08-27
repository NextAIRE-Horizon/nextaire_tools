"""Regression metrics and cross-validation reporting for air-quality models.

Two public entry points:

* :func:`regression_metrics` -- a battery of point-forecast metrics that
  combines the usual statistical scores (MAE, RMSE, :math:`R^2`, correlation)
  with metrics favoured in atmospheric-science model evaluation: the
  *index of agreement* (Willmott's ``d``), the *factor-of-two* fraction
  (``FAC2``), and symmetric MAPE. NaN pairs are dropped before scoring so
  gappy observation series are handled transparently.
* :func:`cross_val_report` -- runs an estimator over a cross-validation
  splitter (e.g. those in :mod:`nextaire_tools.models.splits`) and tabulates the
  held-out metrics per fold, with ``mean`` and ``std`` summary rows.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

from nextaire_tools.exceptions import ConfigurationError, SchemaError
from nextaire_tools.utils.logging import get_logger

__all__ = ["METRIC_NAMES", "cross_val_report", "regression_metrics"]

_LOG = get_logger(__name__)

#: Canonical order of the metrics returned by :func:`regression_metrics`.
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "mse",
    "rmse",
    "r2",
    "mape",
    "smape",
    "wape",
    "nmae",
    "nrmse",
    "bias",
    "pearson_r",
    "spearman_r",
    "index_of_agreement",
    "fac2",
)


def _compute_all(o: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compute every metric on already-aligned, NaN-free arrays."""
    n = o.size
    diff = p - o
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(diff))

    obar = float(np.mean(o))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((o - obar) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # MAPE ignores observations equal to zero (undefined relative error).
    nz = o != 0.0
    mape = float(np.mean(np.abs(diff[nz] / o[nz])) * 100.0) if nz.any() else float("nan")

    # Symmetric MAPE; terms with a zero denominator contribute zero.
    denom = np.abs(p) + np.abs(o)
    with np.errstate(invalid="ignore", divide="ignore"):
        sterm = np.where(denom == 0.0, 0.0, 2.0 * np.abs(diff) / denom)
    smape = float(np.mean(sterm) * 100.0)

    # Weighted absolute percentage error: sum|error| / sum|observed|, in [0, inf).
    abs_o_sum = float(np.sum(np.abs(o)))
    wape = float(np.sum(np.abs(diff)) / abs_o_sum) if abs_o_sum > 0.0 else float("nan")

    # IQR-normalised MAE / RMSE (Q75 - Q25 of the observations). These are the
    # scale-independent errors reported in the hourly-pollutant papers.
    iqr = float(np.subtract(*np.percentile(o, [75, 25])))
    if iqr > 0.0:
        nmae = mae / iqr
        nrmse = rmse / iqr
    else:
        nmae = float("nan")
        nrmse = float("nan")

    if n < 2 or np.std(o) == 0.0 or np.std(p) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(o, p)[0, 1])

    if n < 2:
        spearman = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman = float(spearmanr(o, p)[0])

    # Willmott's index of agreement.
    d_denom = float(np.sum((np.abs(p - obar) + np.abs(o - obar)) ** 2))
    ioa = float(1.0 - ss_res / d_denom) if d_denom > 0.0 else float("nan")

    # Factor-of-two: fraction of predictions within [0.5, 2] x observation.
    if nz.any():
        ratio = p[nz] / o[nz]
        fac2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    else:
        fac2 = float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "nmae": nmae,
        "nrmse": nrmse,
        "bias": bias,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "index_of_agreement": ioa,
        "fac2": fac2,
    }


def regression_metrics(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    *,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute point-forecast regression metrics for air-quality predictions.

    Inputs are flattened, paired element-wise, and any pair with a NaN in
    either array is dropped before scoring.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values. Must have the same length as ``y_true``.
    metrics:
        Optional subset of metric names to return (see :data:`METRIC_NAMES`).
        When ``None`` all metrics are returned.

    Returns
    -------
    dict of str to float
        Metric name to value. Undefined metrics (e.g. correlation on a single
        sample) are returned as ``float('nan')``. Includes:

        ``mae``, ``mse``, ``rmse``
            Absolute / squared error scores.
        ``r2``
            Coefficient of determination.
        ``mape``, ``smape``
            (Symmetric) mean absolute percentage error, in percent.
        ``wape``
            Weighted absolute percentage error, ``sum|error| / sum|observed|``.
        ``nmae``, ``nrmse``
            MAE / RMSE normalised by the inter-quartile range of the
            observations (scale-independent, as reported in the papers).
        ``bias``
            Mean signed error ``mean(pred - true)``.
        ``pearson_r``, ``spearman_r``
            Linear and rank correlation coefficients.
        ``index_of_agreement``
            Willmott's ``d``.
        ``fac2``
            Fraction of predictions within a factor of two of observations.

    Raises
    ------
    SchemaError
        If the inputs differ in length, or no valid pairs remain after NaN
        removal.
    ConfigurationError
        If ``metrics`` names an unknown metric.

    Examples
    --------
    >>> regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])["mae"]
    0.13333333333333336
    >>> sorted(regression_metrics([1, 2], [1, 2], metrics=["mae", "rmse"]))
    ['mae', 'rmse']
    """
    o = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if o.shape[0] != p.shape[0]:
        raise SchemaError(f"y_true and y_pred length mismatch: {o.shape[0]} vs {p.shape[0]}.")

    mask = ~(np.isnan(o) | np.isnan(p))
    o = o[mask]
    p = p[mask]
    if o.size == 0:
        raise SchemaError("No valid (non-NaN) samples remain to compute metrics.")

    result = _compute_all(o, p)

    if metrics is not None:
        requested = list(metrics)
        unknown = [m for m in requested if m not in result]
        if unknown:
            raise ConfigurationError(f"Unknown metric(s): {unknown}. Valid names: {list(result)}.")
        result = {m: result[m] for m in requested}
    return result


def _take(data: object, idx: np.ndarray) -> object:
    """Positionally index ``data`` by integer positions ``idx``."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[idx]
    return np.asarray(data)[idx]


def cross_val_report(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    cv: Any,
    metrics: Iterable[str] | None = None,
    clone_estimator: bool = True,
) -> pd.DataFrame:
    """Evaluate an estimator across cross-validation folds and tabulate metrics.

    For each ``(train, test)`` split produced by ``cv`` the estimator is fitted
    on the training rows and scored on the held-out rows with
    :func:`regression_metrics`. A fresh clone is trained per fold (when
    ``clone_estimator``) so folds never share fitted state.

    Parameters
    ----------
    model:
        A scikit-learn-style regressor exposing ``fit`` and ``predict``.
    X:
        Feature matrix, a :class:`pandas.DataFrame` or NumPy array.
    y:
        Target vector, a :class:`pandas.Series` or NumPy array.
    cv:
        A splitter exposing ``split(X, y)`` (e.g.
        :class:`~nextaire_tools.models.BlockingTimeSeriesSplit`).
    metrics:
        Optional subset of metric names (see :data:`METRIC_NAMES`).
    clone_estimator:
        When ``True`` (default) each fold trains a
        :func:`sklearn.base.clone` of ``model``; otherwise ``model`` is fitted
        in place on every fold.

    Returns
    -------
    pandas.DataFrame
        One row per fold (index ``0, 1, ...``) with a column per metric, plus
        summary rows labelled ``"mean"`` and ``"std"`` computed across the
        folds. The index is named ``"fold"``.

    Raises
    ------
    ConfigurationError
        If ``cv`` yields no folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from nextaire_tools.models.splits import ExpandingWindowSplit
    >>> X = np.arange(40).reshape(-1, 1).astype(float)
    >>> y = X.ravel() * 2.0
    >>> cv = ExpandingWindowSplit(initial_train_size=20, test_size=5)
    >>> report = cross_val_report(LinearRegression(), X, y, cv=cv)
    >>> "mean" in report.index and "rmse" in report.columns
    True
    """
    splits = list(cv.split(X, y))
    if not splits:
        raise ConfigurationError("The cross-validator produced no folds for the given data.")

    # Materialise once: a one-shot iterable would be exhausted after fold 0,
    # silently NaN-ing every subsequent fold.
    metrics = list(metrics) if metrics is not None else None

    rows: list[dict[str, float]] = []
    index: list[int] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        estimator = clone(model) if clone_estimator else model
        estimator.fit(_take(X, train_idx), _take(y, train_idx))
        y_hat = np.asarray(estimator.predict(_take(X, test_idx))).ravel()
        fold_metrics = regression_metrics(_take(y, test_idx), y_hat, metrics=metrics)
        _LOG.debug("fold %d metrics: %s", i, fold_metrics)
        rows.append(fold_metrics)
        index.append(i)

    report = pd.DataFrame(rows, index=index)
    # Summarise across folds *before* appending the summary rows.
    mean_vals = report.mean(numeric_only=True)
    std_vals = report.std(numeric_only=True)
    report.loc["mean"] = mean_vals
    report.loc["std"] = std_vals
    report.index.name = "fold"
    return report
