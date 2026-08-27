# Installation

`nextaire_tools` runs on **Python 3.10+**. The core install is intentionally light —
tabular IO, preprocessing, visualization, and classical scikit-learn modeling —
and the heavier capabilities (deep learning, Copernicus extraction, geospatial
features, Prophet forecasting, national holidays) are opt-in *extras* so you only
pull in what you use.

## Install the core

=== "pip"

    ```bash
    pip install nextaire_tools
    ```

=== "uv"

    ```bash
    uv pip install nextaire_tools
    ```

=== "From source"

    ```bash
    git clone https://github.com/NextAIRE-Horizon/nextaire_tools
    cd nextaire_tools
    pip install -e .
    ```

The core depends on numpy, pandas, scipy, scikit-learn, matplotlib, seaborn,
pyarrow (Parquet IO), and openpyxl (Excel IO). That is enough for
[`load_table`](../api/io.md), every preprocessing step and
[`Pipeline`](../api/preprocessing.md), all of [`nextaire_tools.viz`](../api/viz.md), the
scikit-learn regressors, the time-series splitters, and the metrics.

## Optional extras

Install one or more extras with the usual bracket syntax (quote it so your shell
does not interpret the brackets):

```bash
pip install "nextaire_tools[deep]"                 # one extra
pip install "nextaire_tools[extract,forecast]"     # several at once
pip install "nextaire_tools[all]"                  # every runtime extra
```

| Extra | Pulls in | Unlocks |
| --- | --- | --- |
| *(core)* | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pyarrow, openpyxl | `load_table` / `save_table`, all preprocessing steps + `Pipeline`, `nextaire_tools.viz` plots, `make_regressor` + scikit-learn models, time-series splitters, `regression_metrics` / `cross_val_report` |
| `deep` | PyTorch (`torch`) | `MLPRegressor`, `LSTMRegressor`, `CNNRegressor`, `make_sequences` |
| `extract` | `cdsapi`, `xarray`, `cfgrib`, `eccodes` | Copernicus reanalysis extractors — `ERA5Extractor`, `CAMSExtractor`, `ERA5LandExtractor` — and grid sampling |
| `geo` | `geopandas`, `osmnx`, `shapely` | Geospatial / OpenStreetMap land-use features |
| `forecast` | `prophet` | `ProphetForecaster` |
| `holidays` | `holidays` | the `is_holiday` field in `TemporalFeatures` |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, … | build this documentation site locally |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `build` | run the test suite, lint, type-check |
| `all` | every **runtime** extra above (`deep` + `extract` + `geo` + `forecast` + `holidays`) | everything except the `docs` / `dev` tooling |

!!! note "`all` is runtime-only"
    `nextaire_tools[all]` bundles the runtime extras but **not** `docs` or `dev`.
    Contributors want `pip install -e ".[dev]"` (add `docs` to build the site).

!!! tip "You don't need an extra just to `import`"
    `import nextaire_tools`, `import nextaire_tools.models`, and `import nextaire_tools.extractors` all
    succeed with the core install. An optional dependency is only required when
    you actually *use* the feature that needs it — for example fitting an
    `LSTMRegressor` (needs `torch`) or calling `.extract_to_frames` (needs
    `cdsapi`). When it is missing you get an actionable
    `MissingDependencyError` telling you which extra to install, rather than an
    `ImportError` at startup. See [Key concepts](concepts.md#optional-dependencies-and-require)
    for how this gating works.

## Verifying the install

From Python:

```python
import nextaire_tools

nextaire_tools.__version__
# '0.1.0'
```

The bundled `nextaire_tools` command-line tool reports the version and, with the `info`
subcommand, which optional dependencies are present:

```bash
nextaire_tools --version
# nextaire_tools 0.1.0

nextaire_tools info
```

On a fresh **core-only** install every optional package shows as missing, each
with the exact command to add it:

```text
nextaire_tools 0.1.0

Optional dependencies:
  [-- ] cdsapi     missing  (pip install 'nextaire_tools[extract]')
  [-- ] cfgrib     missing  (pip install 'nextaire_tools[extract]')
  [-- ] geopandas  missing  (pip install 'nextaire_tools[geo]')
  [-- ] holidays   missing  (pip install 'nextaire_tools[holidays]')
  [-- ] osmnx      missing  (pip install 'nextaire_tools[geo]')
  [-- ] prophet    missing  (pip install 'nextaire_tools[forecast]')
  [-- ] torch      missing  (pip install 'nextaire_tools[deep]')
  [-- ] xarray     missing  (pip install 'nextaire_tools[extract]')
```

An installed dependency instead shows `[OK ] … available`. Use `nextaire_tools info` as a
quick environment check before running a workflow that relies on an extra.

## Copernicus credentials (for the `extract` extra)

The [Copernicus extractors](../user-guide/extractors.md) download ERA5 and
ERA5-Land from the **Climate Data Store (CDS)** and CAMS from the **Atmosphere
Data Store (ADS)**. Both require a free account and a **Personal Access Token
(PAT)**. Installing `nextaire_tools[extract]` gets you the client library; you still need
to configure a token.

`nextaire_tools` resolves credentials in this order:

1. the `key=` argument passed to an extractor;
2. the `CDSAPI_KEY` environment variable;
3. the token in a `~/.cdsapirc` file (read by `cdsapi` itself).

### Option A — a `~/.cdsapirc` file

Create `~/.cdsapirc` (in your home directory) with your token:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

Each extractor already points its client at the correct store endpoint
(`ERA5Extractor` and `ERA5LandExtractor` → CDS, `CAMSExtractor` → ADS), so the
`url:` line above is only used when you call `cdsapi` directly. Under the current
unified Copernicus login the same Personal Access Token works for both the CDS
and the ADS, so a single token is usually enough.

### Option B — an environment variable

```bash
export CDSAPI_KEY="<your-personal-access-token>"
```

### Option C — inline

```python
from nextaire_tools.extractors import ERA5Extractor

era5 = ERA5Extractor(key="<your-personal-access-token>")
```

Follow the official setup instructions and accept each dataset's licence before
your first request:

- CDS (ERA5, ERA5-Land): <https://cds.climate.copernicus.eu/how-to-api>
- ADS (CAMS): <https://ads.atmosphere.copernicus.eu/how-to-api>

!!! warning "Large requests are slow"
    A reanalysis request can queue for minutes to hours and stream gigabytes of
    GRIB. Run extraction from a **script or a background job**, not
    interactively, and slice long date ranges into per-month requests. See the
    [extractors guide](../user-guide/extractors.md) and
    [Copernicus data sources](../reference/data-sources.md) for the details.

## Next steps

- [Quickstart](quickstart.md) — a complete load → clean → model walkthrough.
- [Key concepts](concepts.md) — the step / pipeline mental model.
- [API reference](../api/index.md) — every public name, module by module.
