"""Prophet-based forecasting for air-quality time series.

Wraps Facebook Prophet in a small, consistent interface that accepts the data
shapes common in this package -- a two-column ``DataFrame``, a
:class:`~pandas.Series` indexed by time, or explicit ``ds`` / ``y`` array-likes
-- and normalises them to the ``ds`` / ``y`` frame Prophet expects internally.

Prophet is an **optional** dependency (the ``forecast`` extra). It is imported
lazily inside :meth:`ProphetForecaster.fit` via
:func:`nextaire_tools.utils.validation.require`, so importing this module with only the
core dependencies installed always succeeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from nextaire_tools.exceptions import ConfigurationError, NotFittedError, SchemaError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import require

__all__ = ["ProphetForecaster"]

_LOG = get_logger(__name__)

# What Prophet's seasonality flags accept.
SeasonalityArg = bool | str


class ProphetForecaster:
    """Additive time-series forecaster backed by Facebook Prophet.

    Prophet models a series as trend + seasonality + holidays and yields
    calibrated uncertainty intervals, which suits the strong daily/weekly cycles
    seen in urban air-quality data. Internally Prophet requires a DataFrame with
    a datetime column named ``ds`` and a numeric target column named ``y``; this
    wrapper builds that frame for you from several convenient input shapes.

    Parameters
    ----------
    growth:
        Prophet trend model: ``"linear"``, ``"logistic"``, or ``"flat"``.
    yearly_seasonality, weekly_seasonality, daily_seasonality:
        Prophet seasonality switches; ``"auto"``, a bool, or a Fourier order.
    **prophet_kwargs:
        Any additional keyword arguments forwarded verbatim to the ``Prophet``
        constructor (e.g. ``changepoint_prior_scale``, ``interval_width``).

    Attributes
    ----------
    model_ : prophet.Prophet or None
        The fitted Prophet model (``None`` until :meth:`fit` is called).
    last_fit_frame_ : pandas.DataFrame or None
        The ``ds`` / ``y`` frame used for the most recent fit.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> idx = pd.date_range("2021-01-01", periods=240, freq="H")  # doctest: +SKIP
    >>> s = pd.Series(range(240), index=idx, dtype=float)  # doctest: +SKIP
    >>> fc = ProphetForecaster(daily_seasonality=True).fit(s)  # doctest: +SKIP
    >>> out = fc.predict(periods=24, freq="H")  # doctest: +SKIP
    >>> list(out.columns)  # doctest: +SKIP
    ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """

    def __init__(
        self,
        *,
        growth: str = "linear",
        yearly_seasonality: SeasonalityArg = "auto",
        weekly_seasonality: SeasonalityArg = "auto",
        daily_seasonality: SeasonalityArg = "auto",
        **prophet_kwargs: Any,
    ) -> None:
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.prophet_kwargs = dict(prophet_kwargs)
        self.model_: Any | None = None
        self.last_fit_frame_: pd.DataFrame | None = None

    def fit(
        self,
        df: pd.DataFrame | pd.Series | None = None,
        *,
        ds: Sequence[Any] | np.ndarray | pd.Index | None = None,
        y: Sequence[float] | np.ndarray | pd.Series | None = None,
    ) -> ProphetForecaster:
        """Fit the Prophet model.

        Provide the data in exactly one of the supported shapes:

        * ``df`` as a :class:`~pandas.DataFrame` with columns ``ds`` and ``y``;
        * ``df`` as a :class:`~pandas.Series` with a
          :class:`~pandas.DatetimeIndex` (index becomes ``ds``, values ``y``);
        * both ``ds`` and ``y`` as array-likes of equal length.

        Parameters
        ----------
        df:
            A ``ds``/``y`` DataFrame or a time-indexed Series.
        ds:
            Timestamps (used together with ``y`` when ``df`` is ``None``).
        y:
            Target values (used together with ``ds`` when ``df`` is ``None``).

        Returns
        -------
        self

        Raises
        ------
        ConfigurationError
            If no valid combination of arguments is supplied.
        SchemaError
            If the input cannot be coerced to a ``ds``/``y`` frame or has no
            valid rows.
        nextaire_tools.exceptions.MissingDependencyError
            If Prophet (the ``forecast`` extra) is not installed.
        """
        prophet = require("prophet", "forecast", "Prophet forecasting")
        frame = self._to_frame(df, ds, y)

        model = prophet.Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.prophet_kwargs,
        )
        model.fit(frame)
        self.model_ = model
        self.last_fit_frame_ = frame
        _LOG.info("Fitted Prophet on %d observations.", len(frame))
        return self

    def predict(
        self,
        periods: int | None = None,
        freq: str = "H",
        future: pd.DataFrame | pd.DatetimeIndex | Sequence[Any] | None = None,
    ) -> pd.DataFrame:
        """Forecast future (or in-sample) values.

        Parameters
        ----------
        periods:
            Number of future steps of frequency ``freq`` to append after the
            training history. Ignored when ``future`` is given. When both
            ``periods`` and ``future`` are ``None``, predictions are produced on
            the training timestamps only.
        freq:
            Pandas offset alias for the future timestamps (default hourly).
        future:
            Explicit timestamps to predict on: a DataFrame with a ``ds`` column,
            a :class:`~pandas.DatetimeIndex`, or any datetime-like array.

        Returns
        -------
        pandas.DataFrame
            Columns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        SchemaError
            If an explicit ``future`` cannot be coerced to timestamps.
        """
        if self.model_ is None or self.last_fit_frame_ is None:
            raise NotFittedError("This ProphetForecaster is not fitted yet. Call 'fit' first.")

        if future is not None:
            future_frame = self._coerce_future(future)
        elif periods is None:
            future_frame = self.last_fit_frame_[["ds"]].copy()
        else:
            future_frame = self.model_.make_future_dataframe(periods=int(periods), freq=freq)

        forecast = self.model_.predict(future_frame)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # --------------------------------------------------------- internals
    @staticmethod
    def _to_frame(
        df: pd.DataFrame | pd.Series | None,
        ds: Sequence[Any] | np.ndarray | pd.Index | None,
        y: Sequence[float] | np.ndarray | pd.Series | None,
    ) -> pd.DataFrame:
        """Normalise the accepted input shapes to a ``ds``/``y`` DataFrame."""
        if df is not None:
            if isinstance(df, pd.Series):
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise SchemaError("Series input must have a DatetimeIndex to serve as 'ds'.")
                frame = pd.DataFrame(
                    {
                        "ds": pd.to_datetime(df.index),
                        "y": df.to_numpy(dtype=float),
                    }
                )
            elif isinstance(df, pd.DataFrame):
                missing = [c for c in ("ds", "y") if c not in df.columns]
                if missing:
                    raise SchemaError(
                        f"DataFrame is missing required column(s) {missing}. "
                        "Prophet expects columns named 'ds' and 'y'."
                    )
                frame = pd.DataFrame(
                    {
                        "ds": pd.to_datetime(df["ds"]),
                        "y": pd.to_numeric(df["y"], errors="coerce"),
                    }
                )
            else:
                raise SchemaError("df must be a pandas DataFrame (ds, y) or a time-indexed Series.")
        elif ds is not None and y is not None:
            ds_index = pd.to_datetime(pd.Index(ds))
            y_arr = np.asarray(y, dtype=float).ravel()
            if len(ds_index) != len(y_arr):
                raise SchemaError(
                    f"ds and y must have equal length: {len(ds_index)} vs {len(y_arr)}."
                )
            frame = pd.DataFrame({"ds": ds_index, "y": y_arr})
        else:
            raise ConfigurationError(
                "Provide either df (a ds/y DataFrame or a time-indexed Series) or both ds and y."
            )

        frame = frame.dropna(subset=["ds", "y"]).reset_index(drop=True)
        if frame.empty:
            raise SchemaError("No valid (ds, y) rows remain after dropping NaNs.")
        return frame

    @staticmethod
    def _coerce_future(
        future: pd.DataFrame | pd.DatetimeIndex | Sequence[Any],
    ) -> pd.DataFrame:
        """Coerce an explicit ``future`` argument to a single-column ds frame."""
        if isinstance(future, pd.DataFrame):
            if "ds" not in future.columns:
                raise SchemaError("A future DataFrame must contain a 'ds' column of timestamps.")
            return pd.DataFrame({"ds": pd.to_datetime(future["ds"])})
        return pd.DataFrame({"ds": pd.to_datetime(pd.Index(future))})
