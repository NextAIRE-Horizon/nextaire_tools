# Notebooks

Runnable Jupyter notebooks that exercise `nextaire_tools` on synthetic example data — no
network, accounts, or downloads required. They double as smoke tests: each is
committed with its executed outputs and re-runs top to bottom.

| Notebook | What it covers |
|----------|----------------|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | End-to-end: load/save, EDA plots, a cleaning + feature `Pipeline`, and a Random Forest with a leakage-free time-series split. |
| [`02_preprocessing_and_features.ipynb`](02_preprocessing_and_features.ipynb) | Every preprocessing step in turn — missing values (incl. iterative), outliers (incl. `rolling_sigma`), temporal, wind, lag, and correlation-filter transformers — then composed into one pipeline. |
| [`03_reproduce_papers.ipynb`](03_reproduce_papers.ipynb) | Compact interactive versions of the three paper recipes in [`../reproductions/`](../reproductions): RF + metrics (Paper 1), blocked-CV WAPE (Paper 2), NMF + RF + SHAP (Paper 3). |
| [`04_deep_learning_and_forecasting.ipynb`](04_deep_learning_and_forecasting.ipynb) | PyTorch `MLPRegressor` / `LSTMRegressor` / `CNNRegressor` (`[deep]`), and `ProphetForecaster` / `HybridProphetRegressor` (`[forecast]`). Checks for both optional extras and skips gracefully if either is missing. |

## Running

```bash
pip install -e ".[all]"        # or a subset; notebooks skip missing optional deps
jupyter lab                    # then open a notebook, or:

# run headless (also how they are verified):
jupyter nbconvert --to notebook --execute --inplace notebooks/01_quickstart.ipynb
```

The example data is generated on the fly by the helpers in
[`../reproductions/_synthetic.py`](../reproductions/_synthetic.py); swap those for
`nextaire_tools.load_table(...)` of your own files to use real data.
