"""Tests for NMF-based source apportionment (numpy/pandas/sklearn only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.models.source_apportionment import NMFApportionment


def _synthetic_sources(n_samples: int = 120, seed: int = 0) -> pd.DataFrame:
    """Two latent sources mixed into four non-negative pollutant columns."""
    rng = np.random.default_rng(seed)
    W = rng.random((n_samples, 2))
    H = np.array(
        [
            [1.0, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.9, 1.0],
        ]
    )
    V = W @ H + rng.random((n_samples, 4)) * 0.05
    return pd.DataFrame(V, columns=["pm25", "pm10", "no2", "o3"])


def test_fit_transform_shapes_and_names():
    X = _synthetic_sources()
    model = NMFApportionment(n_components=2, random_state=0)
    W = model.fit_transform(X)

    assert isinstance(W, pd.DataFrame)
    assert W.shape == (len(X), 2)
    assert list(W.columns) == ["factor_1", "factor_2"]
    assert W.index.equals(X.index)


def test_components_and_loadings():
    X = _synthetic_sources()
    model = NMFApportionment(n_components=2, random_state=0).fit(X)

    assert model.components_.shape == (2, X.shape[1])
    assert list(model.components_.columns) == list(X.columns)
    assert list(model.components_.index) == ["factor_1", "factor_2"]
    assert model.loadings().equals(model.components_)
    assert list(model.feature_names_in_) == list(X.columns)


def test_transform_matches_fit_transform():
    X = _synthetic_sources()
    model = NMFApportionment(n_components=2, random_state=0)
    W1 = model.fit_transform(X)
    W2 = model.transform(X)
    np.testing.assert_allclose(W1.to_numpy(), W2.to_numpy())


def test_reconstruction_is_reasonable():
    X = _synthetic_sources()
    model = NMFApportionment(n_components=2, random_state=0, max_iter=1000)
    W = model.fit_transform(X).to_numpy()
    H = model.components_.to_numpy()
    # Reconstruct in the scaled space and compare to the scaled input.
    scaled = model.scaler_.transform(X.to_numpy(dtype=float))
    recon = W @ H
    rel_err = np.linalg.norm(scaled - recon) / np.linalg.norm(scaled)
    assert rel_err < 0.2


def test_nan_input_raises():
    X = _synthetic_sources()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ConfigurationError):
        NMFApportionment(n_components=2).fit(X)


def test_negative_input_without_scaling_raises():
    X = _synthetic_sources()
    X.iloc[0, 0] = -1.0
    with pytest.raises(ConfigurationError):
        NMFApportionment(n_components=2, scale=False).fit(X)


def test_negative_input_allowed_when_scaled():
    X = _synthetic_sources()
    X.iloc[0, 0] = -1.0
    # scale=True shifts to [0, 1], so negatives are tolerated.
    model = NMFApportionment(n_components=2, scale=True, random_state=0).fit(X)
    assert model.components_.shape == (2, X.shape[1])
