"""Model-interpretation helpers shared across the papers.

Two families of tools live here:

* :func:`permutation_importance_report` -- a thin, tidy wrapper over
  :func:`sklearn.inspection.permutation_importance` that returns a sorted
  DataFrame (works with any fitted scikit-learn estimator; no extra deps).
* :func:`tree_shap_values` / :func:`shap_importance` -- TreeSHAP explanations for
  tree ensembles. ``shap`` is an **optional** dependency imported lazily, so
  importing this module never requires it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nextaire_tools.utils.logging import get_logger

__all__ = ["permutation_importance_report", "tree_shap_values", "shap_importance"]

_LOG = get_logger(__name__)

_SHAP_HINT = "TreeSHAP requires the optional 'shap' package: pip install shap"


def _feature_names(X: Any) -> list[str]:
    """Return column names for ``X`` (DataFrame columns or ``feature_i``)."""
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    arr = np.asarray(X)
    n = arr.shape[1] if arr.ndim == 2 else 1
    return [f"feature_{i}" for i in range(n)]


def permutation_importance_report(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    n_repeats: int = 5,
    random_state: int = 0,
    scoring: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute permutation importances and return them as a tidy DataFrame.

    Parameters
    ----------
    estimator:
        A fitted scikit-learn-compatible estimator.
    X, y:
        Validation data on which importances are measured. When ``X`` is a
        DataFrame its columns name the features, otherwise ``feature_0 ..``.
    n_repeats:
        Number of shuffles per feature.
    random_state:
        Seed for the permutations.
    scoring:
        Scorer forwarded to :func:`sklearn.inspection.permutation_importance`.
    **kwargs:
        Extra keyword arguments forwarded to the scikit-learn function.

    Returns
    -------
    pandas.DataFrame
        Columns ``["feature", "importance_mean", "importance_std"]`` sorted by
        ``importance_mean`` descending.
    """
    from sklearn.inspection import permutation_importance

    names = _feature_names(X)
    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        **kwargs,
    )
    report = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    report = report.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return report


def _require_shap() -> Any:
    """Import and return the optional :mod:`shap` module."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(_SHAP_HINT) from exc
    return shap


def tree_shap_values(model: Any, X: Any, **kwargs: Any) -> Any:
    """Compute TreeSHAP values for a fitted tree-ensemble ``model``.

    Parameters
    ----------
    model:
        A fitted tree-based model supported by :class:`shap.TreeExplainer`
        (random forest, gradient boosting, XGBoost, ...).
    X:
        Feature matrix to explain.
    **kwargs:
        Extra keyword arguments forwarded to
        :meth:`shap.TreeExplainer.shap_values`.

    Returns
    -------
    The SHAP values as returned by :meth:`shap.TreeExplainer.shap_values`.

    Raises
    ------
    ImportError
        If the optional ``shap`` package is not installed.
    """
    shap = _require_shap()
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X, **kwargs)


def shap_importance(model: Any, X: Any, **kwargs: Any) -> pd.DataFrame:
    """Rank features by mean absolute TreeSHAP value.

    Parameters
    ----------
    model:
        A fitted tree-based model (see :func:`tree_shap_values`).
    X:
        Feature matrix to explain. DataFrame columns name the features.
    **kwargs:
        Forwarded to :func:`tree_shap_values`.

    Returns
    -------
    pandas.DataFrame
        Columns ``["feature", "mean_abs_shap"]`` sorted descending.

    Raises
    ------
    ImportError
        If the optional ``shap`` package is not installed.
    """
    values = tree_shap_values(model, X, **kwargs)
    arr = np.asarray(values)
    # Some explainers return a list (multi-output) or a 3-D array; collapse any
    # trailing output axis by averaging the absolute contributions.
    mean_abs = np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3 else np.abs(arr).mean(axis=0)
    report = pd.DataFrame(
        {"feature": _feature_names(X), "mean_abs_shap": np.asarray(mean_abs).ravel()}
    )
    report = report.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return report
