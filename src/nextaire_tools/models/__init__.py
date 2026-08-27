"""Modeling: time-series cross-validation, metrics, scikit-learn regressors, and
optional PyTorch (MLP/LSTM/CNN) and Prophet models.

The deep-learning and Prophet wrappers import cleanly without ``torch`` / ``prophet``;
those dependencies are only required when a model is actually fitted.
"""

from __future__ import annotations

from nextaire_tools.models.deep import (
    CNNRegressor,
    LSTMRegressor,
    MLPRegressor,
    make_sequences,
)
from nextaire_tools.models.evaluate import cross_val_report, regression_metrics
from nextaire_tools.models.forecast import ProphetForecaster
from nextaire_tools.models.hybrid import HybridProphetRegressor, ProphetFeatures
from nextaire_tools.models.interpret import (
    permutation_importance_report,
    shap_importance,
    tree_shap_values,
)
from nextaire_tools.models.sklearn_models import REGRESSORS, list_regressors, make_regressor
from nextaire_tools.models.source_apportionment import NMFApportionment
from nextaire_tools.models.splits import (
    BlockingTimeSeriesSplit,
    ExpandingWindowSplit,
    SlidingWindowSplit,
    TimeSeriesSplit,
    temporal_train_test_split,
)

__all__ = [
    # splits
    "TimeSeriesSplit",
    "BlockingTimeSeriesSplit",
    "SlidingWindowSplit",
    "ExpandingWindowSplit",
    "temporal_train_test_split",
    # evaluation
    "regression_metrics",
    "cross_val_report",
    # sklearn
    "make_regressor",
    "list_regressors",
    "REGRESSORS",
    # deep
    "make_sequences",
    "MLPRegressor",
    "LSTMRegressor",
    "CNNRegressor",
    # forecast
    "ProphetForecaster",
    "HybridProphetRegressor",
    "ProphetFeatures",
    # source apportionment
    "NMFApportionment",
    # interpretation
    "permutation_importance_report",
    "tree_shap_values",
    "shap_importance",
]
