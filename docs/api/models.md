# `nextaire_tools.models`

Time-series cross-validation, air-quality metrics, a scikit-learn regressor
factory, optional PyTorch deep models, and a Prophet wrapper.

## Cross-validation splitters

::: nextaire_tools.models.BlockingTimeSeriesSplit

::: nextaire_tools.models.SlidingWindowSplit

::: nextaire_tools.models.ExpandingWindowSplit

::: nextaire_tools.models.temporal_train_test_split

## Metrics & reporting

::: nextaire_tools.models.regression_metrics

::: nextaire_tools.models.cross_val_report

## Scikit-learn regressors

::: nextaire_tools.models.make_regressor

::: nextaire_tools.models.list_regressors

## Deep learning (optional `[deep]`)

::: nextaire_tools.models.make_sequences

::: nextaire_tools.models.MLPRegressor

::: nextaire_tools.models.LSTMRegressor

::: nextaire_tools.models.CNNRegressor

## Forecasting (optional `[forecast]`)

::: nextaire_tools.models.ProphetForecaster
