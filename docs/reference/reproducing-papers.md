# Reproducing the papers

`nextaire_tools` exists to make three peer-reviewed air-quality ML studies reproducible
with a small, tested, installable toolkit rather than a pile of one-off scripts.
Each paper has a runnable recipe under
[`reproductions/`](https://github.com/NextAIRE-Horizon/nextaire_tools/tree/main/reproductions)
that rebuilds its data construction, preprocessing, cross-validation, models, and
metrics — **offline**, on synthetic data of the same shape, degrading gracefully
when an optional dependency is absent.

To reproduce the *published numbers*, swap the synthetic generator for a
`nextaire_tools.load_table(...)` of the real data (the Zenodo record for Paper 1;
available on request for Papers 2–3).

```bash
pip install "nextaire_tools[all]"
python reproductions/paper1_petric2024_aaqr.py
python reproductions/paper2_jimenez2024_multitarget.py
python reproductions/paper3_racic2026_source_apportionment.py
```

## The three papers

| # | Paper | Region / task | Core methods |
|---|-------|---------------|--------------|
| 1 | Petrić et al. (2024), *AAQR* 24:230317 | Graz, 5 stations, hourly PM10/NO/NO₂/O₃ | Winsorise → iterative impute → lags → RF / MLP / LSTM / CNN / Prophet / **hybrid** |
| 2 | Jiménez-Navarro et al. (2024), *Results in Engineering* 24:103290 | Graz, 17 targets, 24 h-ahead | Bayesian-ridge impute → lags → **blocked CV** → DT / Lasso / KNN / RF / XGBoost / LSTM |
| 3 | Račić et al. (2026), *Atmospheric Environment: X* 29:100413 | Zagreb, 4 stations, daily PAHs & metals | **NMF** apportionment + per-pollutant RF (log target) + **SHAP** |

## Paper method → nextaire_tools API

| Paper method | nextaire_tools API |
|---|---|
| Rolling 3-day ±4σ winsorisation | [`OutlierHandler(method="rolling_sigma", window=72, sigma=4)`][nextaire_tools.preprocessing.OutlierHandler] |
| Multivariate / Bayesian-ridge imputation | [`MissingValueHandler(strategy="iterative", estimator=...)`][nextaire_tools.preprocessing.MissingValueHandler] |
| Wind direction → x/y, speed from u/v | [`WindDecomposer`][nextaire_tools.preprocessing.WindDecomposer] |
| Calendar + cyclical encodings | [`TemporalFeatures`][nextaire_tools.preprocessing.TemporalFeatures] |
| 12-hour median lag features | [`LagFeatures(windows=[12], agg="median")`][nextaire_tools.preprocessing.LagFeatures] |
| Drop >90 % correlated features | [`CorrelationFilter(threshold=0.9)`][nextaire_tools.preprocessing.CorrelationFilter] |
| Blocked time-series cross-validation | [`BlockingTimeSeriesSplit`][nextaire_tools.models.BlockingTimeSeriesSplit] |
| One-year temporal hold-out | [`temporal_train_test_split`][nextaire_tools.models.temporal_train_test_split] |
| RF / DT / Lasso / KNN / XGBoost | [`make_regressor`][nextaire_tools.models.make_regressor] |
| MLP / LSTM / CNN | [`MLPRegressor`][nextaire_tools.models.MLPRegressor], [`LSTMRegressor`][nextaire_tools.models.LSTMRegressor], [`CNNRegressor`][nextaire_tools.models.CNNRegressor] |
| Prophet, and Prophet→RF hybrid | [`ProphetForecaster`][nextaire_tools.models.ProphetForecaster], [`HybridProphetRegressor`][nextaire_tools.models.HybridProphetRegressor] |
| NMF source apportionment (rank 2) | [`NMFApportionment`][nextaire_tools.models.NMFApportionment] |
| R², nMAE, nRMSE, WAPE, IoA, FAC2 | [`regression_metrics`][nextaire_tools.models.regression_metrics] |
| Permutation importance / TreeSHAP | [`permutation_importance_report`][nextaire_tools.models.permutation_importance_report], [`shap_importance`][nextaire_tools.models.shap_importance] |

## Scope

The recipes reproduce **data construction, preprocessing, cross-validation,
models, and metrics**. Two things are intentionally out of scope: the **Temporal
Selection Layer** (Paper 2's own research contribution — nextaire_tools ships the LSTM/FF
baselines it is compared against, not the layer), and each paper's **exact tuned
hyperparameters** (which live in the supplementary material; the scripts use
sensible defaults so they run quickly).
