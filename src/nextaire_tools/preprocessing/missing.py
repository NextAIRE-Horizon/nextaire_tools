"""Missing-value handling for air-quality time series.

This module provides :class:`MissingValueHandler`, a DataFrame-in / DataFrame-out
:class:`~nextaire_tools.preprocessing.base.BaseStep` that unifies the common strategies for
dealing with gaps in observational data: row dropping, statistical imputation
(mean / median / most-frequent / constant), directional filling
(forward / backward), and time-aware interpolation.

It can additionally

* drop whole columns whose missing fraction exceeds a threshold, and
* emit boolean *missingness indicator* columns so that downstream models can
  learn from the pattern of absence itself.

Notes
-----
The step never mutates the caller's frame; every operation is performed on the
defensive copy supplied by :class:`~nextaire_tools.preprocessing.base.BaseStep`.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.utils.logging import get_logger

__all__ = ["MissingValueHandler"]

_LOG = get_logger(__name__)

_VALID_STRATEGIES = frozenset(
    {
        "drop",
        "mean",
        "median",
        "most_frequent",
        "constant",
        "ffill",
        "bfill",
        "interpolate",
        "iterative",
    }
)

_STATISTIC_STRATEGIES = frozenset({"mean", "median", "most_frequent", "constant"})


class MissingValueHandler(BaseStep):
    """Detect, impute, or remove missing values in selected columns.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Columns to operate on. When ``None`` (default) every column of the input
        frame is used.
    strategy : str, default ``"drop"``
        How to handle missing values. One of

        ``"drop"``
            Drop rows containing a missing value in any selected column.
        ``"mean"`` / ``"median"``
            Impute with the column mean / median (numeric columns only).
        ``"most_frequent"``
            Impute with the column mode.
        ``"constant"``
            Impute with ``fill_value``.
        ``"ffill"`` / ``"bfill"``
            Forward- / backward-fill (respecting ``limit``).
        ``"interpolate"``
            Interpolate. Uses ``method="time"`` when the frame has a
            :class:`~pandas.DatetimeIndex`, otherwise ``method="linear"``.
        ``"iterative"``
            Multivariate imputation: model each numeric column as a function of
            the others, round-robin (scikit-learn ``IterativeImputer``). This is
            the imputation used in the hourly-pollutant papers. The regressor is
            controlled by ``estimator`` (default Bayesian ridge).
    estimator : sklearn estimator, optional
        Regressor used by ``strategy="iterative"``. ``None`` (default) uses
        scikit-learn's own default (:class:`~sklearn.linear_model.BayesianRidge`).
    max_iter : int, default ``10``
        Maximum number of imputation rounds for ``strategy="iterative"``.
    random_state : int, optional
        Seed for the iterative imputer (reproducibility).
    fill_value : object, optional
        Value used when ``strategy="constant"``. Required for that strategy.
    limit : int, optional
        Maximum number of consecutive gaps to fill for ``"ffill"``, ``"bfill"``,
        and ``"interpolate"``. ``None`` means no limit.
    add_indicator : bool, default ``False``
        When ``True``, append an integer ``"<col>__missing"`` column for every
        selected column that contains at least one missing value, computed
        **before** imputation.
    column_missing_threshold : float, optional
        When set, any selected column whose missing fraction strictly exceeds
        this value is dropped entirely. ``None`` (default) disables column
        dropping.

    Attributes
    ----------
    statistics_ : dict
        Mapping of column label to the learned fill value (for the statistical
        strategies only; empty otherwise).
    missing_fraction_ : pandas.Series
        Fraction of missing values per selected column, learned during ``fit``.
    dropped_columns_ : list
        Columns dropped because they exceeded ``column_missing_threshold``.
    indicator_columns_ : list of str
        Names of the missingness-indicator columns that ``transform`` will add.

    Raises
    ------
    ConfigurationError
        If ``strategy`` is not recognised, or ``strategy="constant"`` is used
        without a ``fill_value``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"no2": [1.0, None, 3.0], "o3": [None, 2.0, 3.0]})
    >>> MissingValueHandler(strategy="mean").fit_transform(df)
        no2   o3
    0  1.0  2.5
    1  2.0  2.0
    2  3.0  3.0
    """

    _column_param = "columns"
    _numeric_only = False

    def __init__(
        self,
        columns: ColumnLike | None = None,
        strategy: str = "drop",
        *,
        estimator: object = None,
        max_iter: int = 10,
        random_state: int | None = None,
        fill_value: object = None,
        limit: int | None = None,
        add_indicator: bool = False,
        column_missing_threshold: float | None = None,
    ) -> None:
        self.columns = columns
        self.strategy = strategy
        self.estimator = estimator
        self.max_iter = max_iter
        self.random_state = random_state
        self.fill_value = fill_value
        self.limit = limit
        self.add_indicator = add_indicator
        self.column_missing_threshold = column_missing_threshold

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ConfigurationError(
                f"Unknown strategy {self.strategy!r}. "
                f"Valid strategies: {sorted(_VALID_STRATEGIES)}."
            )
        if self.strategy == "constant" and self.fill_value is None:
            raise ConfigurationError("strategy='constant' requires a non-None 'fill_value'.")

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        selected = self.columns_
        self.missing_fraction_ = X[selected].isna().mean()

        if self.column_missing_threshold is None:
            self.dropped_columns_ = []
        else:
            self.dropped_columns_ = [
                c
                for c in selected
                if float(self.missing_fraction_[c]) > self.column_missing_threshold
            ]
        dropped_set = set(self.dropped_columns_)
        fill_cols = [c for c in selected if c not in dropped_set]

        self.statistics_ = self._learn_statistics(X, fill_cols)

        # Multivariate (iterative) imputation is fitted here on the numeric
        # subset of the fill columns; other strategies leave these unset.
        self._impute_cols: list[Hashable] = []
        self._imputer = None
        if self.strategy == "iterative":
            self._impute_cols = [c for c in fill_cols if pdt.is_numeric_dtype(X[c])]
            if self._impute_cols:
                self._imputer = self._build_iterative_imputer()
                self._imputer.fit(X[self._impute_cols].to_numpy(dtype=float))

        if self.add_indicator:
            # Indicators cover EVERY selected column with missing values,
            # including those dropped by column_missing_threshold — the pattern
            # of absence is most informative for mostly-missing columns.
            self._indicator_cols: list[Hashable] = [
                c for c in selected if float(self.missing_fraction_[c]) > 0.0
            ]
        else:
            self._indicator_cols = []
        self.indicator_columns_ = [f"{c}__missing" for c in self._indicator_cols]

        _LOG.debug(
            "MissingValueHandler fitted: strategy=%s, dropped=%s, indicators=%s",
            self.strategy,
            self.dropped_columns_,
            self.indicator_columns_,
        )

    def _learn_statistics(
        self, X: pd.DataFrame, fill_cols: list[Hashable]
    ) -> dict[Hashable, object]:
        strategy = self.strategy
        if strategy not in _STATISTIC_STRATEGIES:
            return {}

        stats: dict[Hashable, object] = {}
        if strategy in ("mean", "median"):
            for c in fill_cols:
                if pdt.is_numeric_dtype(X[c]):
                    stats[c] = X[c].mean() if strategy == "mean" else X[c].median()
        elif strategy == "most_frequent":
            for c in fill_cols:
                mode = X[c].mode(dropna=True)
                if not mode.empty:
                    stats[c] = mode.iloc[0]
        else:  # constant
            for c in fill_cols:
                stats[c] = self.fill_value
        return stats

    def _build_iterative_imputer(self) -> object:
        """Build a scikit-learn ``IterativeImputer`` (imported lazily)."""
        # IterativeImputer is still behind sklearn's experimental flag.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(
            estimator=self.estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dropped_set = set(self.dropped_columns_)

        # Build missingness indicators BEFORE dropping columns, so indicators for
        # threshold-dropped (mostly-missing) columns are still produced.
        for src, name in zip(self._indicator_cols, self.indicator_columns_):
            if src in X.columns:
                X[name] = X[src].isna().astype(int)

        present_dropped = [c for c in self.dropped_columns_ if c in X.columns]
        if present_dropped:
            X = X.drop(columns=present_dropped)

        present = [c for c in self.columns_ if c in X.columns and c not in dropped_set]
        strategy = self.strategy

        if strategy == "drop":
            if present:
                X = X.dropna(subset=present)
        elif strategy in _STATISTIC_STRATEGIES:
            if self.statistics_:
                X = X.fillna(value=self.statistics_)
        elif strategy in ("ffill", "bfill"):
            if present:
                block = X[present]
                filled = (
                    block.ffill(limit=self.limit)
                    if strategy == "ffill"
                    else block.bfill(limit=self.limit)
                )
                X[present] = filled
        elif strategy == "iterative":
            cols = [c for c in self._impute_cols if c in X.columns]
            if cols and self._imputer is not None:
                X[cols] = self._imputer.transform(X[cols].to_numpy(dtype=float))
        else:  # interpolate
            numeric_present = [c for c in present if pdt.is_numeric_dtype(X[c])]
            if numeric_present:
                method = "time" if isinstance(X.index, pd.DatetimeIndex) else "linear"
                X[numeric_present] = X[numeric_present].interpolate(method=method, limit=self.limit)
        return X

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        Parameters
        ----------
        input_features : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Input names minus dropped columns, plus any indicator columns.
        """
        self._check_is_fitted()
        dropped = set(self.dropped_columns_)
        kept = [c for c in self.feature_names_in_ if c not in dropped]
        return np.asarray(kept + list(self.indicator_columns_), dtype=object)
