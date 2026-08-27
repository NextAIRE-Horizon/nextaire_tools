"""Tests for the scikit-learn regressor factory."""

from __future__ import annotations

import numpy as np
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.models import list_regressors, make_regressor


def test_list_regressors_nonempty():
    names = list_regressors()
    assert "random_forest" in names
    assert "linear" in names


@pytest.mark.parametrize("name", ["linear", "ridge", "random_forest", "hist_gradient_boosting"])
def test_make_and_fit(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = X @ np.array([1.0, 0.0, -1.0, 2.0]) + rng.normal(0, 0.1, 120)
    model = make_regressor(name)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (120,)


def test_unknown_regressor_raises():
    with pytest.raises(ConfigurationError):
        make_regressor("does_not_exist")


def test_params_passthrough():
    model = make_regressor("random_forest", n_estimators=7)
    assert model.get_params()["n_estimators"] == 7
