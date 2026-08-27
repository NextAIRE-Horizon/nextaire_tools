"""Outlier detection and treatment for air-quality time series.

This module provides :class:`OutlierHandler`, a DataFrame-in / DataFrame-out
:class:`~nextaire_tools.preprocessing.base.BaseStep` that detects anomalous values with a
choice of statistical rules (IQR, z-score, modified z-score, quantile bounds) or
a multivariate :class:`sklearn.ensemble.IsolationForest`, and then treats them by
clipping, dropping, masking to ``NaN``, or flagging.

The statistical methods learn a per-column ``(low, high)`` bound during ``fit``
(exposed as :attr:`OutlierHandler.bounds_`); the isolation-forest method learns a
fitted detector instead. Either way, ``fit`` also records the row-level outlier
rate on the training data via :attr:`OutlierHandler.n_outliers_` and
:attr:`OutlierHandler.outlier_fraction_`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import ConfigurationError, SchemaError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.utils.logging import get_logger

if TYPE_CHECKING:  # imported only for typing; sklearn is loaded lazily at fit time
    from sklearn.ensemble import IsolationForest

__all__ = ["OutlierHandler"]

_LOG = get_logger(__name__)

_VALID_METHODS = frozenset(
    {"iqr", "zscore", "modified_zscore", "quantile", "rolling_sigma", "isolation_forest"}
)
_VALID_STRATEGIES = frozenset({"clip", "drop", "nan", "flag"})

# Consistency constant relating the median absolute deviation to the standard
# deviation of a normal distribution (Iglewicz & Hoaglin modified z-score).
_MAD_SCALE = 0.6745


class OutlierHandler(BaseStep):
    """Detect and treat outliers in selected numeric columns.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Numeric columns to operate on. When ``None`` (default) all numeric
        columns are used.
    method : str, default ``"iqr"``
        Detection method. One of ``"iqr"``, ``"zscore"``, ``"modified_zscore"``,
        ``"quantile"`` (per-column bound rules), ``"rolling_sigma"``
        (time-local winsorisation on a rolling window; see ``window`` /
        ``sigma``), or ``"isolation_forest"`` (multivariate, row-level).
    strategy : str, default ``"clip"``
        How to treat detected outliers. One of

        ``"clip"``
            Clip each column to its learned bounds (bound methods only).
        ``"drop"``
            Drop rows flagged as outliers.
        ``"nan"``
            Replace outlying cells with ``NaN``.
        ``"flag"``
            Keep all data and append an integer ``"is_outlier"`` column.
    iqr_factor : float, default ``1.5``
        Multiplier of the inter-quartile range for the ``"iqr"`` method.
    z_threshold : float, default ``3.0``
        Threshold (in standard deviations) for the ``"zscore"`` method.
    mad_threshold : float, default ``3.5``
        Threshold for the ``"modified_zscore"`` method.
    window : int, default ``72``
        Length of the centred rolling window (in rows) for
        ``method="rolling_sigma"``. With hourly data, ``72`` is a three-day
        window — the setting used in the hourly-pollutant papers to remove
        short-lived spikes (e.g. New-Year fireworks).
    sigma : float, default ``4.0``
        Number of within-window standard deviations beyond the within-window
        mean that marks a value as an outlier for ``method="rolling_sigma"``.
    quantiles : tuple of float, default ``(0.01, 0.99)``
        Lower and upper quantiles for the ``"quantile"`` method.
    contamination : float or ``"auto"``, default ``"auto"``
        Expected outlier fraction, forwarded to
        :class:`sklearn.ensemble.IsolationForest`.
    random_state : int, optional
        Random seed forwarded to :class:`~sklearn.ensemble.IsolationForest`.

    Attributes
    ----------
    bounds_ : dict
        Mapping of column label to ``(low, high)`` for bound methods; empty for
        ``"isolation_forest"``.
    detector_ : sklearn.ensemble.IsolationForest or None
        Fitted detector for ``"isolation_forest"``; ``None`` otherwise.
    n_outliers_ : int
        Number of outlier rows detected on the training data.
    outlier_fraction_ : float
        Fraction of outlier rows detected on the training data.

    Raises
    ------
    ConfigurationError
        If ``method`` or ``strategy`` is unknown, if ``quantiles`` is malformed,
        or if ``method="isolation_forest"`` is combined with ``strategy="clip"``.
    SchemaError
        If any selected column is non-numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"pm10": [1.0, 2.0, 3.0, 500.0]})
    >>> OutlierHandler(method="iqr", strategy="clip").fit_transform(df)["pm10"].max() < 500
    True
    """

    _column_param = "columns"
    _numeric_only = True

    def __init__(
        self,
        columns: ColumnLike | None = None,
        method: str = "iqr",
        strategy: str = "clip",
        *,
        iqr_factor: float = 1.5,
        z_threshold: float = 3.0,
        mad_threshold: float = 3.5,
        window: int = 72,
        sigma: float = 4.0,
        quantiles: tuple[float, float] = (0.01, 0.99),
        contamination: float | str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.columns = columns
        self.method = method
        self.strategy = strategy
        self.iqr_factor = iqr_factor
        self.z_threshold = z_threshold
        self.mad_threshold = mad_threshold
        self.window = window
        self.sigma = sigma
        self.quantiles = quantiles
        self.contamination = contamination
        self.random_state = random_state

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ConfigurationError(
                f"Unknown method {self.method!r}. Valid methods: {sorted(_VALID_METHODS)}."
            )
        if self.strategy not in _VALID_STRATEGIES:
            raise ConfigurationError(
                f"Unknown strategy {self.strategy!r}. "
                f"Valid strategies: {sorted(_VALID_STRATEGIES)}."
            )
        if self.method == "isolation_forest" and self.strategy == "clip":
            raise ConfigurationError(
                "strategy='clip' is incompatible with method='isolation_forest'; "
                "use 'drop', 'nan', or 'flag'."
            )
        if self.method == "rolling_sigma":
            if not isinstance(self.window, int) or self.window < 2:
                raise ConfigurationError(
                    f"method='rolling_sigma' requires an integer window >= 2, got {self.window!r}."
                )
            if self.sigma <= 0:
                raise ConfigurationError(
                    f"method='rolling_sigma' requires sigma > 0, got {self.sigma!r}."
                )
        if self.method == "quantile":
            q = self.quantiles
            if len(q) != 2 or not (0.0 <= q[0] < q[1] <= 1.0):
                raise ConfigurationError(
                    f"quantiles must be a (low, high) pair with 0 <= low < high <= 1, got {q!r}."
                )

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        non_numeric = [c for c in self.columns_ if not pdt.is_numeric_dtype(X[c])]
        if non_numeric:
            raise SchemaError(
                f"OutlierHandler requires numeric columns; these are non-numeric: {non_numeric}."
            )

        if self.method == "isolation_forest":
            self.bounds_ = {}
            self.detector_ = self._fit_isolation_forest(X)
            row_mask = self._isolation_forest_row_mask(X)
        elif self.method == "rolling_sigma":
            # Bounds are time-local and recomputed at transform time; nothing
            # global to learn, but we still record the training outlier rate.
            self.bounds_ = {}
            self.detector_ = None
            row_mask = self._rolling_cell_mask(X).any(axis=1)
        else:
            self.bounds_ = {c: self._compute_bounds(X[c]) for c in self.columns_}
            self.detector_ = None
            row_mask = self._cell_mask(X).any(axis=1)

        self.n_outliers_ = int(row_mask.sum())
        self.outlier_fraction_ = float(row_mask.mean()) if len(X) else 0.0
        _LOG.debug(
            "OutlierHandler fitted: method=%s, outliers=%d (%.3f)",
            self.method,
            self.n_outliers_,
            self.outlier_fraction_,
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "isolation_forest":
            row_mask = self._isolation_forest_row_mask(X)
            if self.strategy == "drop":
                return X.loc[~row_mask]
            if self.strategy == "nan":
                X.loc[row_mask, self.columns_] = np.nan
                return X
            X["is_outlier"] = row_mask.astype(int)  # strategy == "flag"
            return X

        if self.method == "rolling_sigma":
            cell_mask = self._rolling_cell_mask(X)
            if self.strategy == "clip":
                for col in self.columns_:
                    low, high = self._rolling_bounds(X[col])
                    X[col] = X[col].clip(lower=low, upper=high)
                return X
            if self.strategy == "drop":
                return X.loc[~cell_mask.any(axis=1)]
            if self.strategy == "nan":
                X[self.columns_] = X[self.columns_].mask(cell_mask)
                return X
            X["is_outlier"] = cell_mask.any(axis=1).astype(int)  # strategy == "flag"
            return X

        cell_mask = self._cell_mask(X)
        if self.strategy == "clip":
            for col in self.columns_:
                low, high = self.bounds_[col]
                X[col] = X[col].clip(lower=low, upper=high)
            return X
        if self.strategy == "drop":
            return X.loc[~cell_mask.any(axis=1)]
        if self.strategy == "nan":
            X[self.columns_] = X[self.columns_].mask(cell_mask)
            return X
        X["is_outlier"] = cell_mask.any(axis=1).astype(int)  # strategy == "flag"
        return X

    # -------------------------------------------------------------- internals
    def _compute_bounds(self, series: pd.Series) -> tuple[float, float]:
        method = self.method
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            return float(q1 - self.iqr_factor * iqr), float(q3 + self.iqr_factor * iqr)
        if method == "zscore":
            mean = series.mean()
            std = series.std()
            return (
                float(mean - self.z_threshold * std),
                float(mean + self.z_threshold * std),
            )
        if method == "modified_zscore":
            median = series.median()
            mad = (series - median).abs().median()
            scale = self.mad_threshold * mad / _MAD_SCALE
            return float(median - scale), float(median + scale)
        # quantile
        low = series.quantile(self.quantiles[0])
        high = series.quantile(self.quantiles[1])
        return float(low), float(high)

    def _rolling_bounds(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return per-row ``(low, high)`` bounds from a centred rolling window."""
        roll = series.rolling(self.window, center=True, min_periods=1)
        mean = roll.mean()
        std = roll.std().fillna(0.0)
        low = mean - self.sigma * std
        high = mean + self.sigma * std
        return low, high

    def _rolling_cell_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-cell outlier mask using time-local rolling-window bounds."""
        mask = pd.DataFrame(False, index=X.index, columns=list(self.columns_))
        for col in self.columns_:
            low, high = self._rolling_bounds(X[col])
            series = X[col]
            # NaN comparisons yield False, so missing values are never flagged.
            mask[col] = (series < low) | (series > high)
        return mask

    def _cell_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a per-cell boolean mask (``True`` where a value is an outlier)."""
        mask = pd.DataFrame(False, index=X.index, columns=list(self.columns_))
        for col in self.columns_:
            low, high = self.bounds_[col]
            series = X[col]
            # NaN comparisons yield False, so missing values are never flagged.
            mask[col] = (series < low) | (series > high)
        return mask

    def _fit_isolation_forest(self, X: pd.DataFrame) -> IsolationForest:
        from sklearn.ensemble import IsolationForest

        complete = X[self.columns_].dropna()
        if complete.empty:
            raise SchemaError(
                "Cannot fit IsolationForest: every row contains a missing value in "
                f"the selected columns {list(self.columns_)}."
            )
        detector = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        detector.fit(complete.to_numpy())
        return detector

    def _isolation_forest_row_mask(self, X: pd.DataFrame) -> pd.Series:
        """Return a row-level boolean mask from the fitted isolation forest.

        Rows with missing values in the selected columns cannot be scored and are
        treated as non-outliers.
        """
        detector = self.detector_
        complete = X[self.columns_].dropna()
        row_mask = pd.Series(False, index=X.index)
        if detector is not None and not complete.empty:
            pred = detector.predict(complete.to_numpy())
            row_mask.loc[complete.index] = pred == -1
        return row_mask

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            The input names, plus ``"is_outlier"`` when ``strategy="flag"``.
        """
        self._check_is_fitted()
        names = list(self.feature_names_in_)
        if self.strategy == "flag":
            names.append("is_outlier")
        return np.asarray(names, dtype=object)
