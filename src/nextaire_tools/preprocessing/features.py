"""Derived-feature engineering for air-quality machine-learning models.

This module bundles three DataFrame-in / DataFrame-out
:class:`~nextaire_tools.preprocessing.base.BaseStep` transformers that reproduce
feature-construction recipes common in the published air-quality ML literature:

* :class:`WindDecomposer` — turn a wind *direction* (in degrees) and/or a wind
  *vector* ``(u, v)`` into stable numeric features, avoiding the 0°/360°
  discontinuity that trips up distance- and gradient-based models.
* :class:`LagFeatures` — append lagged and *causal* rolling-aggregate features
  for time-series columns (e.g. "the median of the twelve most recent
  measurements"), without ever leaking the current row.
* :class:`CorrelationFilter` — drop numeric columns whose absolute pairwise
  Pearson correlation with an already-kept column exceeds a threshold, a classic
  collinearity-reduction step.

All three preserve the frame's (datetime) index, column labels, and dtypes.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import ConfigurationError, SchemaError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import resolve_columns

__all__ = ["WindDecomposer", "LagFeatures", "CorrelationFilter"]

_LOG = get_logger(__name__)

_VALID_AGGS = frozenset({"median", "mean", "min", "max", "std"})


class WindDecomposer(BaseStep):
    """Decompose wind direction and/or a ``(u, v)`` vector into numeric features.

    Wind direction in degrees is circular: 0° and 360° denote the same bearing
    yet sit at opposite ends of the numeric range. Encoding the direction as a
    ``(cos, sin)`` pair removes this discontinuity. Optionally, a wind *speed*
    can be recovered from the eastward/northward components ``(u, v)`` via
    ``sqrt(u**2 + v**2)``.

    Parameters
    ----------
    direction_col : str, optional
        Name of the wind-direction column, measured in **degrees**. When given,
        two columns are emitted: ``"<direction_col>_x" = cos(rad)`` and
        ``"<direction_col>_y" = sin(rad)`` where ``rad = deg * pi / 180``.
    speed_from_uv : tuple of str, optional
        A ``(u_col, v_col)`` pair. When given, a wind-speed column named
        ``speed_out`` is emitted as ``sqrt(u**2 + v**2)``.
    speed_out : str, default ``"wind_speed"``
        Name of the derived wind-speed column.
    drop_original : bool, default ``False``
        When ``True``, drop the source ``direction_col`` / ``u`` / ``v`` columns
        after deriving the new features.

    Attributes
    ----------
    feature_names_out_ : list of str
        Names of the newly generated columns, in deterministic emission order.

    Raises
    ------
    ConfigurationError
        If neither ``direction_col`` nor ``speed_from_uv`` is given, or if
        ``speed_from_uv`` is not a two-element ``(u, v)`` pair.
    SchemaError
        If any referenced column is absent from the input frame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"wd": [0.0, 90.0, 180.0], "u": [1.0, 0.0, -1.0], "v": [0.0, 1.0, 0.0]})
    >>> out = WindDecomposer(direction_col="wd", speed_from_uv=("u", "v")).fit_transform(df)
    >>> sorted(c for c in out.columns if c not in df.columns)
    ['wd_x', 'wd_y', 'wind_speed']
    """

    _column_param = None
    _numeric_only = False

    def __init__(
        self,
        direction_col: str | None = None,
        *,
        speed_from_uv: tuple[str, str] | None = None,
        speed_out: str = "wind_speed",
        drop_original: bool = False,
    ) -> None:
        self.direction_col = direction_col
        self.speed_from_uv = speed_from_uv
        self.speed_out = speed_out
        self.drop_original = drop_original

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if self.direction_col is None and self.speed_from_uv is None:
            raise ConfigurationError(
                "WindDecomposer requires at least one of direction_col or speed_from_uv to be set."
            )
        if self.speed_from_uv is not None and len(self.speed_from_uv) != 2:
            raise ConfigurationError(
                f"speed_from_uv must be a (u_col, v_col) pair, got {self.speed_from_uv!r}."
            )

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        required = self._referenced_columns()
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise SchemaError(
                f"WindDecomposer references columns absent from the frame: {missing}. "
                f"Available columns: {list(X.columns)}"
            )
        self.dropped_columns_ = self._columns_to_drop()
        self.feature_names_out_ = self._build_feature_names()
        _LOG.debug(
            "WindDecomposer fitted: +%d features, -%d columns",
            len(self.feature_names_out_),
            len(self.dropped_columns_),
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new: dict[str, pd.Series] = {}
        if self.direction_col is not None:
            rad = X[self.direction_col].astype(float) * np.pi / 180.0
            new[f"{self.direction_col}_x"] = np.cos(rad)
            new[f"{self.direction_col}_y"] = np.sin(rad)
        if self.speed_from_uv is not None:
            u_col, v_col = self.speed_from_uv
            u = X[u_col].astype(float)
            v = X[v_col].astype(float)
            new[self.speed_out] = np.sqrt(u**2 + v**2)

        new_df = pd.DataFrame(new, index=X.index)[self.feature_names_out_]
        kept = X.drop(columns=list(self.dropped_columns_))
        return pd.concat([kept, new_df], axis=1)

    # -------------------------------------------------------------- internals
    def _referenced_columns(self) -> list[Hashable]:
        cols: list[Hashable] = []
        if self.direction_col is not None:
            cols.append(self.direction_col)
        if self.speed_from_uv is not None:
            cols.extend(self.speed_from_uv)
        return cols

    def _columns_to_drop(self) -> list[Hashable]:
        if not self.drop_original:
            return []
        # Order-preserving de-duplication of the referenced source columns.
        seen: set[Hashable] = set()
        out: list[Hashable] = []
        for c in self._referenced_columns():
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.direction_col is not None:
            names.append(f"{self.direction_col}_x")
            names.append(f"{self.direction_col}_y")
        if self.speed_from_uv is not None:
            names.append(self.speed_out)
        return names

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Surviving input columns (source columns removed when
            ``drop_original=True``) followed by the generated feature names.
        """
        self._check_is_fitted()
        dropped = set(self.dropped_columns_)
        kept = [c for c in self.feature_names_in_ if c not in dropped]
        return np.asarray(kept + list(self.feature_names_out_), dtype=object)


class LagFeatures(BaseStep):
    """Append lagged and causal rolling-aggregate features for time series.

    For each selected column ``c`` this step can emit:

    * **Lags** — ``"<c>_lag{k}" = c.shift(k)`` for each ``k`` in ``lags``.
    * **Rolling aggregates** — ``"<c>_roll{w}_{agg}"`` computed over the ``w``
      *preceding* rows as ``c.shift(1).rolling(w, min_periods=1).agg(agg)``. The
      leading ``shift(1)`` makes the window strictly causal: the current row is
      never included, so no target-adjacent information leaks into the feature.

    The frame is assumed to be sorted in time order; new columns are appended and
    the original columns are retained.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Numeric columns to lag. When ``None`` (default) all numeric columns are
        used.
    lags : sequence of int, default ``()``
        Positive shift amounts. Each ``k`` yields ``"<c>_lag{k}"`` per column.
    windows : sequence of int, default ``()``
        Rolling-window sizes. Each ``w`` yields ``"<c>_roll{w}_{agg}"`` per
        column, aggregated over the ``w`` preceding rows.
    agg : str, default ``"median"``
        Rolling aggregation. One of ``"median"``, ``"mean"``, ``"min"``,
        ``"max"``, ``"std"``.

    Attributes
    ----------
    feature_names_out_ : list of str
        Names of the newly generated columns, in deterministic emission order.

    Raises
    ------
    ConfigurationError
        If both ``lags`` and ``windows`` are empty, if ``agg`` is unsupported, or
        if any lag/window is not a positive integer.
    SchemaError
        If any selected column is non-numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"no2": [1.0, 2.0, 3.0, 4.0]})
    >>> out = LagFeatures(lags=(1,), windows=(2,), agg="mean").fit_transform(df)
    >>> list(out.columns)
    ['no2', 'no2_lag1', 'no2_roll2_mean']
    """

    _column_param = "columns"
    _numeric_only = True

    def __init__(
        self,
        columns: ColumnLike | None = None,
        *,
        lags: Sequence[int] = (),
        windows: Sequence[int] = (),
        agg: str = "median",
    ) -> None:
        self.columns = columns
        self.lags = lags
        self.windows = windows
        self.agg = agg

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if not self.lags and not self.windows:
            raise ConfigurationError(
                "LagFeatures requires at least one of lags or windows to be non-empty."
            )
        if self.agg not in _VALID_AGGS:
            raise ConfigurationError(
                f"Unknown agg {self.agg!r}. Valid aggregations: {sorted(_VALID_AGGS)}."
            )
        bad_lags = [k for k in self.lags if not isinstance(k, (int, np.integer)) or k < 1]
        if bad_lags:
            raise ConfigurationError(f"lags must be positive integers, got {bad_lags}.")
        bad_windows = [w for w in self.windows if not isinstance(w, (int, np.integer)) or w < 1]
        if bad_windows:
            raise ConfigurationError(f"windows must be positive integers, got {bad_windows}.")

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        from pandas.api import types as pdt

        non_numeric = [c for c in self.columns_ if not pdt.is_numeric_dtype(X[c])]
        if non_numeric:
            raise SchemaError(
                f"LagFeatures requires numeric columns; these are non-numeric: {non_numeric}."
            )
        self.feature_names_out_ = self._build_feature_names()
        _LOG.debug("LagFeatures fitted: %d new features", len(self.feature_names_out_))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_out_:
            return X

        new: dict[str, pd.Series] = {}
        for col in self.columns_:
            series = X[col]
            for k in self.lags:
                new[f"{col}_lag{k}"] = series.shift(int(k))
            for w in self.windows:
                # shift(1) excludes the current row, keeping the window causal.
                rolled = series.shift(1).rolling(int(w), min_periods=1).agg(self.agg)
                new[f"{col}_roll{w}_{self.agg}"] = rolled

        new_df = pd.DataFrame(new, index=X.index)[self.feature_names_out_]
        return pd.concat([X, new_df], axis=1)

    # -------------------------------------------------------------- internals
    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        for col in self.columns_:
            for k in self.lags:
                names.append(f"{col}_lag{k}")
            for w in self.windows:
                names.append(f"{col}_roll{w}_{self.agg}")
        return names

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            The input column names followed by the generated feature names.
        """
        self._check_is_fitted()
        names = list(self.feature_names_in_) + list(self.feature_names_out_)
        return np.asarray(names, dtype=object)


class CorrelationFilter(BaseStep):
    """Drop numeric columns that are highly collinear with a kept column.

    Among the candidate columns, the step keeps the first column and then walks
    the remaining candidates in order, dropping any whose absolute Pearson
    correlation with an already-kept candidate exceeds ``threshold``. Columns
    outside the candidate set are always kept, and columns named in ``protect``
    are never dropped.

    Parameters
    ----------
    threshold : float, default ``0.9``
        Absolute-correlation cut-off in ``(0, 1]``. A candidate is dropped when
        its ``|corr|`` with any previously-kept candidate exceeds this value.
    columns : str or sequence of str, optional
        Candidate columns to consider. When ``None`` (default) all numeric
        columns are candidates. Non-candidate columns are always kept.
    protect : str or sequence of str, optional
        Columns that must never be dropped (e.g. the target).

    Attributes
    ----------
    dropped_columns_ : list of hashable
        Candidate columns removed during ``fit``, in the order they were dropped.
    kept_columns_ : list of hashable
        Candidate columns retained during ``fit``.

    Raises
    ------
    ConfigurationError
        If ``threshold`` is not in ``(0, 1]``.
    SchemaError
        If any candidate column is non-numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [3.0, 1.0, 2.0]})
    >>> out = CorrelationFilter(threshold=0.95).fit_transform(df)
    >>> "b" in out.columns  # b is perfectly correlated with a
    False
    """

    _column_param = "columns"
    _numeric_only = True

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        columns: ColumnLike | None = None,
        protect: ColumnLike | None = None,
    ) -> None:
        self.threshold = threshold
        self.columns = columns
        self.protect = protect

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if not (0.0 < self.threshold <= 1.0):
            raise ConfigurationError(
                f"threshold must satisfy 0 < threshold <= 1, got {self.threshold!r}."
            )

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        from pandas.api import types as pdt

        non_numeric = [c for c in self.columns_ if not pdt.is_numeric_dtype(X[c])]
        if non_numeric:
            raise SchemaError(
                f"CorrelationFilter requires numeric candidate columns; these are "
                f"non-numeric: {non_numeric}."
            )

        protected = set(resolve_columns(X, self.protect) if self.protect is not None else [])

        candidates = list(self.columns_)
        abs_corr = X[candidates].corr().abs() if candidates else pd.DataFrame()

        kept: list[Hashable] = []
        dropped: list[Hashable] = []
        for col in candidates:
            if col in protected:
                kept.append(col)
                continue
            # Drop when correlated above threshold with any already-kept candidate.
            collinear = any(
                kept_col in abs_corr.index and float(abs_corr.loc[col, kept_col]) > self.threshold
                for kept_col in kept
            )
            if collinear:
                dropped.append(col)
            else:
                kept.append(col)

        self.kept_columns_ = kept
        self.dropped_columns_ = dropped
        _LOG.debug(
            "CorrelationFilter fitted: dropped %d of %d candidates",
            len(dropped),
            len(candidates),
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.dropped_columns_:
            return X
        return X.drop(columns=list(self.dropped_columns_))

    # -------------------------------------------------------------- internals
    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            The input column names with the dropped columns removed.
        """
        self._check_is_fitted()
        dropped = set(self.dropped_columns_)
        names = [c for c in self.feature_names_in_ if c not in dropped]
        return np.asarray(names, dtype=object)
