"""Tests for the hybrid Prophet + regressor model.

Prophet is an optional dependency; the fit/predict tests are gated with
``pytest.importorskip("prophet")`` and skip cleanly when it is absent.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.models.hybrid import HybridProphetRegressor, ProphetFeatures

prophet_missing = importlib.util.find_spec("prophet") is None
requires_prophet = pytest.mark.skipif(prophet_missing, reason="prophet not installed")


def _daily_series(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    trend = np.linspace(0, 5, n)
    weekly = np.sin(2 * np.pi * idx.dayofweek / 7)
    y = trend + weekly + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"temp": np.linspace(10, 20, n)}, index=idx)
    return X, pd.Series(y, index=idx)


def test_predict_before_fit_raises():
    X, _ = _daily_series()
    from nextaire_tools.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        HybridProphetRegressor().predict(X)


def test_fit_requires_datetime():
    # No DatetimeIndex and no 'ds' column.
    X = pd.DataFrame({"temp": [1.0, 2.0, 3.0]})
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ConfigurationError):
        HybridProphetRegressor().fit(X, y)


@requires_prophet
def test_hybrid_fit_predict_adds_prophet_features():
    from sklearn.ensemble import RandomForestRegressor

    X, y = _daily_series()
    model = HybridProphetRegressor(
        base_estimator=RandomForestRegressor(n_estimators=50, random_state=0),
    ).fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()

    # Prophet forecast columns were generated and appended.
    assert "yhat" in model.prophet_feature_names_
    assert "yhat_lower" in model.prophet_feature_names_
    assert "yhat_upper" in model.prophet_feature_names_
    # add_components defaults to True -> trend is present.
    assert "trend" in model.prophet_feature_names_
    assert list(model.feature_names_in_) == ["temp"]


@requires_prophet
def test_hybrid_predicts_out_of_sample():
    X, y = _daily_series(n=120)
    model = HybridProphetRegressor().fit(X, y)

    future_idx = pd.date_range("2021-05-01", periods=14, freq="D")
    X_future = pd.DataFrame({"temp": np.linspace(15, 18, 14)}, index=future_idx)
    preds = model.predict(X_future)
    assert preds.shape == (14,)
    assert np.isfinite(preds).all()


@requires_prophet
def test_prophet_features_transformer():
    X, y = _daily_series()
    frame = X.copy()
    frame["target"] = y.to_numpy()

    pf = ProphetFeatures("target")
    feats = pf.fit_transform(frame)
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == len(frame)
    assert feats.index.equals(frame.index)
    assert "yhat" in feats.columns
