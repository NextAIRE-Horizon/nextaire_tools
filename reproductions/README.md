# Reproductions

Runnable recipes that rebuild the *methodology* of the three papers behind
`nextaire_tools` (see [`../papers/`](../papers/)) end-to-end. Each script:

- runs **offline** on synthetic data of the same shape as the paper's dataset
  (`_synthetic.py`) — no network, no accounts, no optional deps required for the
  core path;
- is organised as an explicit pipeline where **every step names the nextaire_tools
  building block** it uses, so you can read it as a map from paper → API;
- degrades gracefully when an optional dependency is missing (deep models,
  Prophet, XGBoost, SHAP), printing how to install it.

To reproduce the *published numbers*, swap the `make_*` synthetic generator for a
`nextaire_tools.load_table(...)` of the real data (public Zenodo record for Paper 1;
available on request for Papers 2–3).

```bash
pip install -e ".[all]"          # or a subset: [deep], [forecast], [boost], [shap]
python reproductions/paper1_petric2024_aaqr.py
python reproductions/paper2_jimenez2024_multitarget.py
python reproductions/paper3_racic2026_source_apportionment.py
```

## Paper → nextaire_tools building blocks

| Paper method | nextaire_tools API |
|---|---|
| Rolling 3-day ±4σ winsorisation | `OutlierHandler(method="rolling_sigma", window=72, sigma=4)` |
| Multivariate / Bayesian-Ridge imputation | `MissingValueHandler(strategy="iterative", estimator=...)` |
| Wind direction → x/y, speed from u/v | `WindDecomposer` |
| Calendar + cyclical encodings | `TemporalFeatures` |
| 12-hour median lag features | `LagFeatures(windows=[12], agg="median")` |
| Drop >90 % correlated features | `CorrelationFilter(threshold=0.9)` |
| Blocked time-series cross-validation | `BlockingTimeSeriesSplit` |
| One-year temporal hold-out | `temporal_train_test_split` |
| RF / DT / Lasso / KNN / XGBoost | `make_regressor(name, ...)` |
| MLP / LSTM / CNN | `MLPRegressor`, `LSTMRegressor`, `CNNRegressor` (`[deep]`) |
| Prophet, and Prophet→RF hybrid | `ProphetForecaster`, `HybridProphetRegressor` (`[forecast]`) |
| NMF source apportionment (rank 2) | `NMFApportionment` |
| R², nMAE, nRMSE, WAPE, IoA, FAC2 | `regression_metrics` |
| Permutation importance / TreeSHAP | `permutation_importance_report`, `shap_importance` (`[shap]`) |

## What is and isn't reproduced

These recipes reproduce the **data construction, preprocessing, cross-validation,
models, and metrics**. Two things are intentionally out of scope:

- **The Temporal Selection Layer (TSL)** from Paper 2 is that paper's own
  research contribution, not a general-purpose block; nextaire_tools ships the LSTM/FF
  baselines it is compared against, not the layer itself.
- **Exact hyperparameters** live in each paper's supplementary material; the
  scripts use sensible defaults / small grids so they run quickly.
