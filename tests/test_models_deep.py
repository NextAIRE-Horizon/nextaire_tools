"""Tests for PyTorch deep regressors (skipped if torch is not installed)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from nextaire_tools.models import make_sequences

torch_missing = importlib.util.find_spec("torch") is None
requires_torch = pytest.mark.skipif(torch_missing, reason="torch not installed")


def test_make_sequences_shapes():
    X = np.arange(100).reshape(-1, 2).astype(float)  # 50 timesteps, 2 features
    Xs, ys = make_sequences(X, np.arange(50.0), window=5, horizon=1)
    assert Xs.ndim == 3
    assert Xs.shape[1:] == (5, 2)
    assert Xs.shape[0] == ys.shape[0]


@requires_torch
@pytest.mark.deep
def test_mlp_fit_predict():
    from nextaire_tools.models import MLPRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.1, 200)
    model = MLPRegressor(hidden_sizes=(16,), epochs=5, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (200,)
    assert np.isfinite(pred).all()


@requires_torch
@pytest.mark.deep
def test_lstm_fit_predict_alignment():
    from nextaire_tools.models import LSTMRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = np.cumsum(rng.normal(size=n)) * 0.01
    model = LSTMRegressor(window=10, hidden_size=8, epochs=3, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert len(pred) == n  # aligned to input rows


@requires_torch
@pytest.mark.deep
def test_cnn_fit_predict():
    from nextaire_tools.models import CNNRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    model = CNNRegressor(window=10, channels=(8,), epochs=3, random_state=0)
    model.fit(X, y)
    assert len(model.predict(X)) == n
