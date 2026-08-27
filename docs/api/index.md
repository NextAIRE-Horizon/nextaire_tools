# API reference

This section is generated automatically from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/). Every public class and function
in `nextaire_tools` is documented here with its full signature, parameters, and examples.

## Package layout

| Module | Purpose |
|--------|---------|
| [`nextaire_tools.io`](io.md) | Read/write CSV, Excel, and Parquet. |
| [`nextaire_tools.preprocessing`](preprocessing.md) | Missing values, outliers, temporal features, scaling, and pipelines. |
| [`nextaire_tools.extractors`](extractors.md) | ERA5 / CAMS / ERA5-Land extraction from the Copernicus data stores. |
| [`nextaire_tools.viz`](viz.md) | EDA, outlier, and model-evaluation figures. |
| [`nextaire_tools.models`](models.md) | Time-series CV, metrics, scikit-learn factory, deep models, and Prophet. |
| [`nextaire_tools.utils` & `nextaire_tools.exceptions`](utils.md) | Validation helpers, logging, and the exception hierarchy. |

## Top-level namespace

The most-used names are re-exported on the package root:

```python
import nextaire_tools

nextaire_tools.load_table          # nextaire_tools.io.load_table
nextaire_tools.save_table          # nextaire_tools.io.save_table
nextaire_tools.Pipeline            # nextaire_tools.preprocessing.Pipeline
nextaire_tools.make_pipeline       # nextaire_tools.preprocessing.make_pipeline
nextaire_tools.MissingValueHandler
nextaire_tools.OutlierHandler
nextaire_tools.TemporalFeatures
nextaire_tools.Scaler
nextaire_tools.enable_logging      # nextaire_tools.utils.logging.enable_logging
```
