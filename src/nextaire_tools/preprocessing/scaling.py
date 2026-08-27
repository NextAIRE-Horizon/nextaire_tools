"""Feature scaling for air-quality time series.

This module provides :class:`Scaler`, a thin, DataFrame-preserving wrapper around
the scikit-learn scalers (:class:`~sklearn.preprocessing.StandardScaler`,
:class:`~sklearn.preprocessing.MinMaxScaler`,
:class:`~sklearn.preprocessing.RobustScaler`,
:class:`~sklearn.preprocessing.MaxAbsScaler`).

Unlike the bare scikit-learn transformers — which return a NumPy array and
discard column names and the index — :class:`Scaler` scales only the selected
numeric columns in place within a copy, leaving every other column, dtype, and
the (datetime) index untouched. It also exposes :meth:`Scaler.inverse_transform`
to map scaled values back to their original units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import ColumnNotFoundError, ConfigurationError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import check_dataframe

__all__ = ["Scaler"]

_LOG = get_logger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


class Scaler(BaseStep):
    """Scale selected numeric columns, preserving the DataFrame structure.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Numeric columns to scale. When ``None`` (default) all numeric columns
        are scaled.
    method : str, default ``"standard"``
        Scaling method. One of ``"standard"`` (zero mean, unit variance),
        ``"minmax"`` (scaled to ``[0, 1]``), ``"robust"`` (median / IQR), or
        ``"maxabs"`` (scaled by the maximum absolute value).

    Attributes
    ----------
    scaler_ : sklearn.base.TransformerMixin
        The fitted scikit-learn scaler.

    Raises
    ------
    ConfigurationError
        If ``method`` is not one of the supported values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"no2": [1.0, 2.0, 3.0], "flag": ["a", "b", "c"]})
    >>> scaler = Scaler(columns=["no2"], method="standard")
    >>> scaled = scaler.fit_transform(df)
    >>> restored = scaler.inverse_transform(scaled)
    >>> bool((restored["no2"].round(6) == df["no2"]).all())
    True
    """

    _column_param = "columns"
    _numeric_only = True

    def __init__(self, columns: ColumnLike | None = None, method: str = "standard") -> None:
        self.columns = columns
        self.method = method

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if self.method not in _SCALERS:
            raise ConfigurationError(
                f"Unknown method {self.method!r}. Valid methods: {sorted(_SCALERS)}."
            )

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        self.scaler_ = _SCALERS[self.method]()
        if self.columns_:
            self.scaler_.fit(X[self.columns_].to_numpy())
        _LOG.debug("Scaler fitted: method=%s on %d column(s)", self.method, len(self.columns_))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_:
            X[self.columns_] = self.scaler_.transform(X[self.columns_].to_numpy())
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Invert the scaling of the fitted columns.

        Parameters
        ----------
        X : pandas.DataFrame
            A frame containing the fitted columns in their scaled representation.

        Returns
        -------
        pandas.DataFrame
            A new frame with the fitted columns mapped back to original units;
            all other columns and the index are left unchanged.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ColumnNotFoundError
            If any fitted column is absent from ``X``.
        """
        self._check_is_fitted()
        Xdf = check_dataframe(X, copy=True)
        if self.columns_:
            missing = [c for c in self.columns_ if c not in Xdf.columns]
            if missing:
                raise ColumnNotFoundError(
                    f"Columns required for inverse_transform are missing: {missing}. "
                    f"Available columns: {list(Xdf.columns)}"
                )
            Xdf[self.columns_] = self.scaler_.inverse_transform(Xdf[self.columns_].to_numpy())
        return Xdf

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names (unchanged by scaling).

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            The input column names.
        """
        self._check_is_fitted()
        return np.asarray(self.feature_names_in_, dtype=object)
