"""Factory and registry for scikit-learn regressors used in air-quality ML.

A single :func:`make_regressor` entry point builds any of the estimators the
package supports from a short string name, so pipelines and CLIs can select a
model by configuration without importing scikit-learn classes directly. The
:data:`REGRESSORS` mapping is the source of truth; :func:`list_regressors`
enumerates the available names.

Tree-based ensembles (random forest, extra trees, gradient boosting) default to
``random_state=0`` for reproducibility -- matching the author's published
random-forest hourly-forecasting setup -- unless the caller overrides it. The
seed is only injected into estimators whose constructor actually accepts it.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.utils.logging import get_logger

__all__ = ["REGRESSORS", "list_regressors", "make_regressor"]

_LOG = get_logger(__name__)


def _xgboost_regressor(**params: Any) -> BaseEstimator:
    """Construct an :class:`xgboost.XGBRegressor`, importing ``xgboost`` lazily.

    XGBoost is an optional dependency (``pip install "nextaire_tools[boost]"`` or
    ``pip install xgboost``); the import is deferred to call time so the rest of
    the registry works without it.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised only without xgboost
        raise ConfigurationError(
            "The 'xgboost' regressor requires the optional 'xgboost' package. "
            'Install it with: pip install "nextaire_tools[boost]" (or: pip install xgboost).'
        ) from exc
    return XGBRegressor(**params)


#: Registry mapping a short name to a zero-/keyword-argument estimator factory.
REGRESSORS: dict[str, Callable[..., BaseEstimator]] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "xgboost": _xgboost_regressor,
    "svr": SVR,
    "knn": KNeighborsRegressor,
    "mlp": MLPRegressor,
}

# Tree/forest models that default to a fixed seed for reproducibility.
_RANDOM_STATE_DEFAULTS: frozenset[str] = frozenset(
    {
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "xgboost",
    }
)


def _accepts(factory: Callable[..., BaseEstimator], param: str) -> bool:
    """Return whether ``factory``'s constructor accepts a keyword ``param``.

    A factory that only declares ``**kwargs`` (e.g. :func:`_xgboost_regressor`,
    which forwards everything to the wrapped estimator) is treated as accepting
    any keyword, since its own signature can't otherwise reveal what the
    wrapped constructor supports.
    """
    try:
        parameters = signature(factory).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtins without a signature
        return False
    if param in parameters:
        return True
    return any(p.kind is Parameter.VAR_KEYWORD for p in parameters.values())


def list_regressors() -> list[str]:
    """Return the sorted list of registered regressor names.

    Returns
    -------
    list of str
        Names accepted by :func:`make_regressor`.

    Examples
    --------
    >>> "random_forest" in list_regressors()
    True
    """
    return sorted(REGRESSORS)


def make_regressor(name: str = "random_forest", **params: Any) -> BaseEstimator:
    """Construct a scikit-learn regressor by name.

    Parameters
    ----------
    name:
        A key from :data:`REGRESSORS` (case-insensitive). Defaults to
        ``"random_forest"``.
    **params:
        Keyword arguments forwarded to the estimator constructor. They override
        any package default (including the tree-model ``random_state``).

    Returns
    -------
    sklearn.base.BaseEstimator
        An unfitted estimator instance.

    Raises
    ------
    ConfigurationError
        If ``name`` is not registered, or the supplied ``params`` are invalid
        for the chosen estimator.

    Examples
    --------
    >>> reg = make_regressor("random_forest", n_estimators=50)
    >>> reg.get_params()["random_state"]
    0
    >>> make_regressor("ridge", alpha=2.0).alpha
    2.0
    """
    key = name.lower() if isinstance(name, str) else name
    if key not in REGRESSORS:
        valid = ", ".join(sorted(REGRESSORS))
        raise ConfigurationError(f"Unknown regressor {name!r}. Valid names: {valid}.")

    factory = REGRESSORS[key]
    if (
        key in _RANDOM_STATE_DEFAULTS
        and "random_state" not in params
        and _accepts(factory, "random_state")
    ):
        params["random_state"] = 0

    try:
        estimator = factory(**params)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid parameters for regressor {name!r}: {exc}") from exc

    _LOG.debug("Built regressor %r: %s", key, type(estimator).__name__)
    return estimator
