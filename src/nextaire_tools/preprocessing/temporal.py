"""Calendar and cyclical time-feature engineering for air-quality series.

This module provides :class:`TemporalFeatures`, the flagship
:class:`~nextaire_tools.preprocessing.base.BaseStep` for turning a timestamp — either the
frame's :class:`~pandas.DatetimeIndex` or a named column — into a rich set of
predictive features:

* **Calendar fields** such as ``hour``, ``dayofweek``, ``month``, ``season``, and
  boolean flags like ``is_weekend`` or ``is_holiday``.
* **Cyclical encodings** (``sin`` / ``cos`` pairs) that give periodic fields a
  smooth, wrap-around representation — e.g. hour 23 sits next to hour 0 — which
  is far friendlier to distance- and gradient-based models than a raw integer.

National-holiday features require the optional :mod:`holidays` dependency, which
is imported lazily only when ``"is_holiday"`` is requested.

Notes
-----
Cyclical encoding maps an integer field ``v`` to
``sin(2*pi*(v - base)/period)`` and the matching cosine, using period/base pairs
chosen so that each field wraps at its natural boundary (see
``_CYCLICAL_PERIODS``).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from nextaire_tools.exceptions import ColumnNotFoundError, ConfigurationError, SchemaError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import require

__all__ = ["TemporalFeatures"]

_LOG = get_logger(__name__)

# Calendar fields that ``add`` understands.
_SUPPORTED_FIELDS = frozenset(
    {
        "year",
        "quarter",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "hour",
        "minute",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "season",
        "is_holiday",
    }
)

# Fields that can be cyclically encoded, mapped to ``(base, period)``.
_CYCLICAL_PERIODS: dict[str, tuple[float, float]] = {
    "hour": (0.0, 24.0),
    "minute": (0.0, 60.0),
    "dayofweek": (0.0, 7.0),
    "month": (1.0, 12.0),
    "quarter": (1.0, 4.0),
    "dayofyear": (1.0, 365.25),
    "weekofyear": (1.0, 52.1775),
    "day": (1.0, 31.0),
}


class TemporalFeatures(BaseStep):
    """Derive calendar and cyclical features from a datetime index or column.

    Parameters
    ----------
    time_col : str, optional
        Name of the timestamp column. When given, the column is parsed with
        :func:`pandas.to_datetime` and **kept** in the output. When ``None``
        (default) the frame's :class:`~pandas.DatetimeIndex` is used.
    add : sequence of str, default ``("hour", "dayofweek", "month", "dayofyear", "is_weekend")``
        Calendar fields to append as raw columns. Supported fields: ``year``,
        ``quarter``, ``month``, ``day``, ``dayofweek``, ``dayofyear``,
        ``weekofyear``, ``hour``, ``minute``, ``is_weekend``, ``is_month_start``,
        ``is_month_end``, ``season`` (1=DJF, 2=MAM, 3=JJA, 4=SON), ``is_holiday``.
    cyclical : sequence of str, default ``("hour", "dayofweek", "month", "dayofyear")``
        Fields to encode as ``sin`` / ``cos`` pairs. A field may be encoded
        cyclically even if it is not in ``add``. Supported fields: ``hour``,
        ``minute``, ``dayofweek``, ``month``, ``quarter``, ``dayofyear``,
        ``weekofyear``, ``day``.
    drop_raw_cyclical : bool, default ``False``
        When ``True``, do not emit the raw integer column for fields that are
        cyclically encoded (the ``sin`` / ``cos`` pair is still emitted).
    holidays_country : str, optional
        ISO country code (e.g. ``"HR"``, ``"US"``) used to build the holiday
        calendar. Required when ``"is_holiday"`` is in ``add``.
    prefix : str, default ``""``
        String prepended to every generated column name.

    Attributes
    ----------
    feature_names_out_ : list of str
        Names of the newly generated columns, in deterministic emission order.

    Raises
    ------
    ConfigurationError
        If ``add`` or ``cyclical`` contains an unsupported field, or if
        ``"is_holiday"`` is requested without ``holidays_country``.
    SchemaError
        If no datetime index and no ``time_col`` are available.
    ColumnNotFoundError
        If ``time_col`` is given but absent from the frame.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=3, freq="h", name="t")
    >>> df = pd.DataFrame({"no2": [1.0, 2.0, 3.0]}, index=idx)
    >>> out = TemporalFeatures(add=("hour",), cyclical=("hour",)).fit_transform(df)
    >>> sorted(c for c in out.columns if c != "no2")
    ['hour', 'hour_cos', 'hour_sin']
    """

    _column_param = None
    _numeric_only = False

    def __init__(
        self,
        time_col: Hashable | None = None,
        *,
        add: Sequence[str] = ("hour", "dayofweek", "month", "dayofyear", "is_weekend"),
        cyclical: Sequence[str] = ("hour", "dayofweek", "month", "dayofyear"),
        drop_raw_cyclical: bool = False,
        holidays_country: str | None = None,
        prefix: str = "",
    ) -> None:
        self.time_col = time_col
        self.add = add
        self.cyclical = cyclical
        self.drop_raw_cyclical = drop_raw_cyclical
        self.holidays_country = holidays_country
        self.prefix = prefix

    # ------------------------------------------------------------------ hooks
    def _validate_params(self) -> None:
        bad_add = [f for f in self.add if f not in _SUPPORTED_FIELDS]
        if bad_add:
            raise ConfigurationError(
                f"Unsupported calendar field(s) in add={list(self.add)}: {bad_add}. "
                f"Supported: {sorted(_SUPPORTED_FIELDS)}."
            )
        bad_cyc = [f for f in self.cyclical if f not in _CYCLICAL_PERIODS]
        if bad_cyc:
            raise ConfigurationError(
                f"Unsupported cyclical field(s) in cyclical={list(self.cyclical)}: "
                f"{bad_cyc}. Cyclically encodable: {sorted(_CYCLICAL_PERIODS)}."
            )
        if "is_holiday" in self.add and self.holidays_country is None:
            raise ConfigurationError(
                "add includes 'is_holiday' but holidays_country is None; pass an ISO "
                "country code such as holidays_country='US'."
            )

    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        # Validate that a usable timestamp is present (raises early on failure).
        self._resolve_time(X)
        if "is_holiday" in self.add:
            require("holidays", "holidays", "holiday features")
        self.feature_names_out_ = self._build_feature_names()
        _LOG.debug("TemporalFeatures fitted: %d new features", len(self.feature_names_out_))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_out_:
            return X

        t = self._resolve_time(X)
        new: dict[str, pd.Series] = {}
        cyclical_set = set(self.cyclical)

        for field in self.add:
            if self.drop_raw_cyclical and field in cyclical_set:
                continue
            new[f"{self.prefix}{field}"] = self._compute_field(field, t)

        for field in self.cyclical:
            base, period = _CYCLICAL_PERIODS[field]
            values = self._compute_field(field, t).astype(float)
            angle = 2.0 * np.pi * (values - base) / period
            new[f"{self.prefix}{field}_sin"] = np.sin(angle)
            new[f"{self.prefix}{field}_cos"] = np.cos(angle)

        new_df = pd.DataFrame(new, index=X.index)[self.feature_names_out_]
        return pd.concat([X, new_df], axis=1)

    # -------------------------------------------------------------- internals
    def _build_feature_names(self) -> list[str]:
        cyclical_set = set(self.cyclical)
        names: list[str] = []
        for field in self.add:
            if self.drop_raw_cyclical and field in cyclical_set:
                continue
            names.append(f"{self.prefix}{field}")
        for field in self.cyclical:
            names.append(f"{self.prefix}{field}_sin")
            names.append(f"{self.prefix}{field}_cos")
        return names

    def _resolve_time(self, df: pd.DataFrame) -> pd.Series:
        """Resolve a datetime :class:`~pandas.Series` aligned to ``df.index``."""
        if self.time_col is not None:
            if self.time_col not in df.columns:
                raise ColumnNotFoundError(
                    f"time_col={self.time_col!r} not found. Available columns: {list(df.columns)}"
                )
            t = pd.to_datetime(df[self.time_col], errors="coerce")
            if t.isna().all():
                raise SchemaError(f"Column {self.time_col!r} could not be parsed as datetime.")
            return t

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            di = idx
        elif pdt.is_datetime64_any_dtype(idx):
            di = pd.DatetimeIndex(idx)
        else:
            raise SchemaError(
                "TemporalFeatures requires a DatetimeIndex or a time_col. No datetime "
                "index found; pass time_col=<column name> or set a DatetimeIndex on "
                "the frame first."
            )
        return pd.Series(di, index=df.index)

    def _compute_field(self, field: str, t: pd.Series) -> pd.Series:
        dt = t.dt
        if field == "year":
            return dt.year
        if field == "quarter":
            return dt.quarter
        if field == "month":
            return dt.month
        if field == "day":
            return dt.day
        if field == "dayofweek":
            return dt.dayofweek
        if field == "dayofyear":
            return dt.dayofyear
        if field == "weekofyear":
            return dt.isocalendar().week.astype(int)
        if field == "hour":
            return dt.hour
        if field == "minute":
            return dt.minute
        if field == "is_weekend":
            return (dt.dayofweek >= 5).astype(int)
        if field == "is_month_start":
            return dt.is_month_start.astype(int)
        if field == "is_month_end":
            return dt.is_month_end.astype(int)
        if field == "season":
            return ((dt.month % 12 + 3) // 3).astype(int)
        # is_holiday
        return self._compute_holidays(t)

    def _compute_holidays(self, t: pd.Series) -> pd.Series:
        holidays_mod = require("holidays", "holidays", "holiday features")
        years = sorted(int(y) for y in t.dt.year.dropna().unique())
        calendar = holidays_mod.country_holidays(self.holidays_country, years=years)
        holiday_dates = set(calendar.keys())
        flags = [d in holiday_dates for d in t.dt.date]
        return pd.Series(flags, index=t.index, dtype=int)

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
