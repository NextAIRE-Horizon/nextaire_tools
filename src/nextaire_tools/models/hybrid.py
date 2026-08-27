"""Hybrid Prophet + regressor model for air-quality forecasting.

This module implements the "hybrid" approach in which a
:class:`~nextaire_tools.models.forecast.ProphetForecaster` is fitted on the target series
and its in-sample forecast (``yhat`` / ``yhat_lower`` / ``yhat_upper``, and
optionally the trend and seasonal components) is appended as extra features to a
classical scikit-learn regressor -- a Random Forest by default. Prophet captures
the strong trend and daily/weekly seasonality of urban pollutant series, and the
downstream estimator learns the residual, feature-driven structure on top.

Prophet is an **optional** dependency (the ``forecast`` extra). It is imported
lazily inside :meth:`HybridProphetRegressor.fit` / :meth:`ProphetFeatures.fit`
(reusing :class:`~nextaire_tools.models.forecast.ProphetForecaster`, which imports it via
:func:`nextaire_tools.utils.validation.require`), so importing this module with only the
core dependencies installed always succeeds.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from nextaire_tools.exceptions import ConfigurationError, NotFittedError
from nextaire_tools.models.forecast import ProphetForecaster
from nextaire_tools.models.sklearn_models import make_regressor
from nextaire_tools.utils.logging import get_logger

__all__ = ["HybridProphetRegressor", "ProphetFeatures"]

_LOG = get_logger(__name__)

# Always-present Prophet forecast columns used as features.
_FORECAST_COLUMNS: tuple[str, ...] = ("yhat", "yhat_lower", "yhat_upper")
# Component columns appended when ``add_components`` is enabled (kept only when
# actually produced by the fitted Prophet model).
_COMPONENT_COLUMNS: tuple[str, ...] = ("trend", "weekly", "yearly", "daily")


def _extract_ds(
    X: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Return ``(timestamps, feature_frame)`` from ``X``.

    Timestamps come from a ``ds`` column when present, otherwise from a
    :class:`~pandas.DatetimeIndex`. The returned feature frame is ``X`` with any
    ``ds`` column removed (the datetime is not itself a model feature).

    Raises
    ------
    ConfigurationError
        If ``X`` is not a DataFrame or exposes no usable datetime.
    """
    if not isinstance(X, pd.DataFrame):
        raise ConfigurationError(f"X must be a pandas.DataFrame, got {type(X).__name__!r}.")
    if "ds" in X.columns:
        ds = pd.DatetimeIndex(pd.to_datetime(X["ds"]))
        features = X.drop(columns=["ds"])
    elif isinstance(X.index, pd.DatetimeIndex):
        ds = X.index
        features = X
    else:
        raise ConfigurationError(
            "X needs a usable datetime: pass a DataFrame with a DatetimeIndex or "
            "a 'ds' column of timestamps."
        )
    if ds.isna().any():
        raise ConfigurationError("Timestamps contain unparseable / missing values.")
    return ds, features


def _select_feature_columns(forecast: pd.DataFrame, add_components: bool) -> list[str]:
    """List the Prophet output columns used as features (order-stable)."""
    cols = [c for c in _FORECAST_COLUMNS if c in forecast.columns]
    if add_components:
        cols += [c for c in _COMPONENT_COLUMNS if c in forecast.columns]
    return cols


class ProphetFeatures:
    """Turn a time-indexed target into Prophet forecast feature columns.

    A small, composable transformer: fit it on a DataFrame that carries the
    target as a named column, and :meth:`transform` returns *only* the Prophet
    forecast columns (``yhat`` / ``yhat_lower`` / ``yhat_upper`` and, when
    ``add_components`` is set, the trend and seasonal components) aligned to the
    input's index. Useful for building a custom pipeline where the augmented
    matrix is assembled by hand.

    Parameters
    ----------
    target:
        Name of the column holding the target series at ``fit`` time.
    prophet_kwargs:
        Extra keyword arguments forwarded to
        :class:`~nextaire_tools.models.forecast.ProphetForecaster`.
    add_components:
        Also emit Prophet trend / seasonal component columns when available.

    Attributes
    ----------
    forecaster_ : nextaire_tools.models.forecast.ProphetForecaster
        The fitted Prophet wrapper (``None`` until :meth:`fit`).
    prophet_feature_names_ : list of str
        Names of the emitted feature columns.
    """

    def __init__(
        self,
        target: Hashable,
        *,
        prophet_kwargs: dict[str, Any] | None = None,
        add_components: bool = True,
    ) -> None:
        self.target = target
        self.prophet_kwargs = prophet_kwargs
        self.add_components = add_components
        self.forecaster_: ProphetForecaster | None = None
        self.prophet_feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: object = None) -> ProphetFeatures:
        """Fit Prophet on ``X[target]`` indexed by ``X``'s timestamps.

        Raises
        ------
        ConfigurationError
            If ``X`` lacks the target column or a usable datetime.
        nextaire_tools.exceptions.MissingDependencyError
            If Prophet (the ``forecast`` extra) is not installed.
        """
        ds, _ = _extract_ds(X)
        if self.target not in X.columns:
            raise ConfigurationError(
                f"target column {self.target!r} not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        y_vals = pd.to_numeric(X[self.target], errors="coerce").to_numpy(dtype=float)
        forecaster = ProphetForecaster(**(self.prophet_kwargs or {}))
        forecaster.fit(ds=ds, y=y_vals)
        self.forecaster_ = forecaster
        # Determine the emitted columns from an in-sample forecast.
        insample = self._raw_forecast(ds)
        self.prophet_feature_names_ = _select_feature_columns(insample, self.add_components)
        _LOG.info("ProphetFeatures fitted; emits %d columns.", len(self.prophet_feature_names_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the Prophet feature columns aligned to ``X``'s index."""
        if self.forecaster_ is None or self.prophet_feature_names_ is None:
            raise NotFittedError(
                "This ProphetFeatures instance is not fitted yet. Call 'fit' first."
            )
        ds, _ = _extract_ds(X)
        forecast = self._raw_forecast(ds)
        out = forecast.reindex(columns=self.prophet_feature_names_).fillna(0.0)
        out.index = X.index
        return out

    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit then transform in one call."""
        return self.fit(X, y).transform(X)

    def _raw_forecast(self, ds: pd.DatetimeIndex) -> pd.DataFrame:
        """Run the fitted Prophet model on ``ds`` and return its full output."""
        assert self.forecaster_ is not None and self.forecaster_.model_ is not None
        future = pd.DataFrame({"ds": pd.to_datetime(ds)})
        return self.forecaster_.model_.predict(future)


class HybridProphetRegressor(RegressorMixin, BaseEstimator):
    """Random-forest (or any regressor) augmented with Prophet forecasts.

    Fitting proceeds in two stages: a Prophet model is fitted on the target
    series, its in-sample forecast columns are concatenated to the feature
    matrix, and a clone of ``base_estimator`` is fitted on the augmented matrix.
    :meth:`predict` regenerates the Prophet columns for the requested timestamps
    -- which may be out-of-sample -- and calls the fitted estimator, so future
    forecasts work as long as the timestamps are supplied.

    Parameters
    ----------
    base_estimator:
        The downstream regressor. When ``None`` a
        ``RandomForestRegressor(n_estimators=400, random_state=0)`` is built via
        :func:`~nextaire_tools.models.sklearn_models.make_regressor`.
    prophet_kwargs:
        Extra keyword arguments forwarded to
        :class:`~nextaire_tools.models.forecast.ProphetForecaster`.
    add_components:
        Also append Prophet trend / seasonal components as features.

    Attributes
    ----------
    estimator_ : sklearn.base.BaseEstimator
        The fitted downstream regressor.
    features_ : ProphetFeatures
        The fitted Prophet feature generator.
    feature_names_in_ : numpy.ndarray
        The non-datetime input feature columns seen at ``fit``.
    prophet_feature_names_ : list of str
        The Prophet feature columns appended to the matrix.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> idx = pd.date_range("2021-01-01", periods=120, freq="D")  # doctest: +SKIP
    >>> X = pd.DataFrame({"temp": range(120)}, index=idx)  # doctest: +SKIP
    >>> y = pd.Series(range(120), index=idx, dtype=float)  # doctest: +SKIP
    >>> model = HybridProphetRegressor().fit(X, y)  # doctest: +SKIP
    >>> preds = model.predict(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        *,
        prophet_kwargs: dict[str, Any] | None = None,
        add_components: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.prophet_kwargs = prophet_kwargs
        self.add_components = add_components

    # --------------------------------------------------------------- API
    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float] | np.ndarray | pd.Series,
    ) -> HybridProphetRegressor:
        """Fit Prophet on ``y`` and the downstream estimator on the augmented ``X``.

        Parameters
        ----------
        X:
            DataFrame with a :class:`~pandas.DatetimeIndex` (or a ``ds`` column)
            and, optionally, additional feature columns.
        y:
            Target series / array aligned row-for-row with ``X``.

        Returns
        -------
        self

        Raises
        ------
        ConfigurationError
            If ``X`` lacks a usable datetime, or ``X`` and ``y`` mismatch.
        nextaire_tools.exceptions.MissingDependencyError
            If Prophet (the ``forecast`` extra) is not installed.
        """
        ds, features = _extract_ds(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.shape[0] != len(features):
            raise ConfigurationError(
                f"X and y length mismatch: {len(features)} vs {y_arr.shape[0]}."
            )

        self.feature_names_in_ = np.asarray(features.columns, dtype=object)

        # Fit the Prophet feature generator on a frame carrying the target.
        target_name = "__hybrid_target__"
        prophet_frame = features.copy()
        prophet_frame[target_name] = y_arr
        prophet_frame.index = ds
        self.features_ = ProphetFeatures(
            target_name,
            prophet_kwargs=self.prophet_kwargs,
            add_components=self.add_components,
        ).fit(prophet_frame)
        self.prophet_feature_names_ = list(self.features_.prophet_feature_names_ or [])

        augmented = self._augment(features, ds)
        estimator = self._make_estimator()
        estimator.fit(augmented.to_numpy(dtype=float), y_arr)
        self.estimator_ = estimator
        _LOG.info(
            "Fitted HybridProphetRegressor on %d rows, %d features (+%d Prophet).",
            len(augmented),
            len(self.feature_names_in_),
            len(self.prophet_feature_names_),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for ``X`` (in-sample or future timestamps).

        Parameters
        ----------
        X:
            DataFrame with a datetime (index or ``ds`` column) and the same
            feature columns seen at ``fit``.

        Returns
        -------
        numpy.ndarray
            1-D array of length ``len(X)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        ds, features = _extract_ds(X)
        augmented = self._augment(features, ds)
        return np.asarray(self.estimator_.predict(augmented.to_numpy(dtype=float)))

    # --------------------------------------------------------- internals
    def _make_estimator(self) -> BaseEstimator:
        if self.base_estimator is None:
            return make_regressor("random_forest", n_estimators=400)
        return clone(self.base_estimator)

    def _augment(self, features: pd.DataFrame, ds: pd.DatetimeIndex) -> pd.DataFrame:
        """Concatenate the fitted Prophet feature columns onto ``features``."""
        base = features.reindex(columns=list(self.feature_names_in_))
        base = base.apply(pd.to_numeric, errors="coerce")
        # Build a ds-indexed frame so ProphetFeatures can featurise the timestamps.
        prophet_input = base.copy()
        prophet_input.index = ds
        prophet_cols = self.features_.transform(prophet_input)
        prophet_cols.index = base.index
        return pd.concat([base, prophet_cols], axis=1)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before 'predict'."
            )
