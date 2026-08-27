"""Tests for the paper-reproduction building blocks added on top of the core.

Covers the rolling-sigma winsoriser and iterative imputer (Papers 1 & 2), the
IQR-normalised / WAPE metrics (Papers 1 & 2), and the decision-tree / xgboost
regressor registrations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.models import make_regressor
from nextaire_tools.models.evaluate import regression_metrics
from nextaire_tools.preprocessing import MissingValueHandler, OutlierHandler


# --------------------------------------------------------------------------- #
# OutlierHandler: rolling-sigma winsorisation
# --------------------------------------------------------------------------- #
def test_rolling_sigma_clips_local_spike(aq_df: pd.DataFrame) -> None:
    handler = OutlierHandler(
        columns=["pm10"], method="rolling_sigma", strategy="clip", window=72, sigma=4.0
    )
    out = handler.fit_transform(aq_df)
    # The injected 500.0 spike at row 50 must be pulled far down…
    assert out["pm10"].iloc[50] < 500.0
    # …while a typical value stays essentially unchanged.
    assert out["pm10"].iloc[400] == pytest.approx(aq_df["pm10"].iloc[400], rel=1e-6)
    assert handler.n_outliers_ >= 1


def test_rolling_sigma_nan_and_flag(aq_df: pd.DataFrame) -> None:
    nanned = OutlierHandler(
        columns=["pm10"], method="rolling_sigma", strategy="nan", window=48, sigma=4.0
    ).fit_transform(aq_df)
    assert np.isnan(nanned["pm10"].iloc[50])

    flagged = OutlierHandler(
        columns=["pm10"], method="rolling_sigma", strategy="flag", window=48, sigma=4.0
    ).fit_transform(aq_df)
    assert "is_outlier" in flagged.columns
    assert flagged["is_outlier"].iloc[50] == 1


def test_rolling_sigma_rejects_bad_window() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ConfigurationError):
        OutlierHandler(method="rolling_sigma", window=1).fit(df)
    with pytest.raises(ConfigurationError):
        OutlierHandler(method="rolling_sigma", sigma=0).fit(df)


def test_rolling_sigma_preserves_index(aq_df: pd.DataFrame) -> None:
    out = OutlierHandler(method="rolling_sigma", strategy="clip").fit_transform(aq_df)
    assert out.index.equals(aq_df.index)
    assert list(out.columns) == list(aq_df.columns)


# --------------------------------------------------------------------------- #
# MissingValueHandler: iterative (multivariate) imputation
# --------------------------------------------------------------------------- #
def test_iterative_imputer_fills_all_gaps(aq_df: pd.DataFrame) -> None:
    out = MissingValueHandler(strategy="iterative", random_state=0).fit_transform(aq_df)
    assert not out.isna().any().any()
    assert out.shape == aq_df.shape
    assert out.index.equals(aq_df.index)


def test_iterative_imputer_custom_estimator(aq_df: pd.DataFrame) -> None:
    from sklearn.linear_model import BayesianRidge

    out = MissingValueHandler(
        strategy="iterative", estimator=BayesianRidge(), max_iter=5, random_state=0
    ).fit_transform(aq_df)
    assert not out["no2"].isna().any()


# --------------------------------------------------------------------------- #
# Metrics: WAPE, nMAE, nRMSE
# --------------------------------------------------------------------------- #
def test_new_metrics_present_and_sane() -> None:
    rng = np.random.default_rng(0)
    o = 20 + 10 * rng.random(200)
    p = o + rng.normal(0, 1.0, 200)
    m = regression_metrics(o, p)
    for name in ("wape", "nmae", "nrmse"):
        assert name in m
        assert np.isfinite(m[name])
        assert m[name] >= 0.0
    # Perfect prediction => zero normalised errors.
    perfect = regression_metrics(o, o)
    assert perfect["wape"] == pytest.approx(0.0)
    assert perfect["nmae"] == pytest.approx(0.0)
    assert perfect["nrmse"] == pytest.approx(0.0)


def test_nmae_equals_mae_over_iqr() -> None:
    o = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    p = o + 1.0  # constant error of 1
    m = regression_metrics(o, p)
    iqr = np.subtract(*np.percentile(o, [75, 25]))
    assert m["nmae"] == pytest.approx(1.0 / iqr)


# --------------------------------------------------------------------------- #
# Regressor registry: decision_tree + xgboost
# --------------------------------------------------------------------------- #
def test_decision_tree_registered() -> None:
    est = make_regressor("decision_tree", max_depth=4)
    assert type(est).__name__ == "DecisionTreeRegressor"
    assert est.get_params()["random_state"] == 0


def test_xgboost_registered_or_clear_error() -> None:
    pytest.importorskip("xgboost")
    est = make_regressor("xgboost", n_estimators=10)
    assert type(est).__name__ == "XGBRegressor"
