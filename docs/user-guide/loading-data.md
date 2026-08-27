# Loading data

`nextaire_tools` reads CSV, Excel, and Parquet through a single entry point,
[`load_table`](../api/io.md), and writes them back with
[`save_table`](../api/io.md). Both infer the on-disk format from the file
extension, so the rest of your workflow never has to care whether the data came
from a spreadsheet or a Parquet file.

## The running example

Every page in this guide uses the same shape of data: an **hourly,
datetime-indexed** frame of pollutant concentrations (`no2`, `o3`, `pm10`).
Here is a small synthetic version you can paste into a REPL:

```python
import numpy as np
import pandas as pd

idx = pd.date_range("2024-01-01", periods=720, freq="h", name="timestamp")
rng = np.random.default_rng(0)
t = np.arange(len(idx))
df = pd.DataFrame(
    {
        "no2": 20 + 10 * np.sin(t * 2 * np.pi / 24) + rng.normal(0, 3, len(idx)),
        "o3": 40 + 15 * np.cos(t * 2 * np.pi / 24) + rng.normal(0, 4, len(idx)),
        "pm10": 15 + rng.gamma(2.0, 3.0, len(idx)),
    },
    index=idx,
)
```

In practice you load this frame from a file instead of building it by hand.

## `load_table` basics

```python
from nextaire_tools import load_table

df = load_table("station.csv")
```

The format is chosen from the suffix. The supported extensions are exposed as
`nextaire_tools.io.SUPPORTED_SUFFIXES`:

| Kind | Extensions | Underlying reader |
| --- | --- | --- |
| CSV / delimited text | `.csv`, `.txt`, `.tsv`, `.dat` | `pandas.read_csv` |
| Excel | `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.ods` | `pandas.read_excel` |
| Parquet | `.parquet`, `.pq`, `.parq` | `pandas.read_parquet` |

Any extra keyword arguments are forwarded to the underlying pandas reader, so
anything pandas supports is available:

```python
df = load_table("station.csv", na_values=["-999", "NA"], dtype={"no2": "float32"})
```

!!! note "Errors are explicit"
    `load_table` raises `FileNotFoundError` if the path does not exist and a
    `nextaire_tools.exceptions.SchemaError` for an unsupported extension (e.g. `.json`).

## Getting a `DatetimeIndex`

Most of `nextaire_tools` — temporal features, time-aware interpolation, seasonality
plots, and the model splitters — expects a sorted
[`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html).
Pass `time_col` to parse a timestamp column with `pandas.to_datetime`, and
`set_time_index=True` to move it into the index:

```python
df = load_table("station.csv", time_col="timestamp", set_time_index=True)

df.index          # DatetimeIndex, sorted ascending
"timestamp" in df.columns   # False — it became the index
```

`time_col` and `set_time_index` are independent:

- `time_col="timestamp"` alone parses the column to datetime but leaves it as a
  column.
- `set_time_index=True` alone promotes an **already datetime-like** index to a
  sorted `DatetimeIndex`.
- Together they parse the column, set it as the index, and sort.

!!! tip "Timestamps that fail to parse become `NaT`"
    Parsing uses `errors="coerce"`, so unparseable strings turn into `NaT`
    rather than raising. Inspect `df.index.isna().sum()` after loading if you
    suspect a malformed timestamp column.

## Selecting a subset of columns

`columns` restricts what is loaded. For Parquet this is **pushed down** to the
reader (only those columns are read from disk); for CSV and Excel it is applied
after reading:

=== "Parquet (pushdown)"

    ```python
    # Only these column chunks are read from disk.
    df = load_table("obs.parquet", columns=["timestamp", "no2", "o3"])
    ```

=== "CSV / Excel (post-read)"

    ```python
    df = load_table("station.csv", columns=["timestamp", "no2"])
    list(df.columns)   # ['timestamp', 'no2']
    ```

Requesting a column that is not present raises `SchemaError` (for CSV/Excel)
listing the missing labels.

You can combine `columns` with the datetime handling — the subset is resolved
first, then the time column is promoted to the index:

```python
df = load_table(
    "obs.parquet",
    columns=["timestamp", "no2", "o3", "pm10"],
    time_col="timestamp",
    set_time_index=True,
)
list(df.columns)   # ['no2', 'o3', 'pm10']  — timestamp is now the index
```

## Excel sheets and text delimiters

`sheet_name` chooses the worksheet for Excel files (by name or zero-based
position); it is ignored for other formats.

```python
df = load_table("monitoring.xlsx", sheet_name="Zagreb-1")
```

!!! warning "One sheet at a time"
    `load_table` returns a single `DataFrame`. Asking pandas for multiple sheets
    (e.g. `sheet_name=None`, which returns a dict) raises `SchemaError`. Load
    each sheet with a separate call.

`sep` sets the field delimiter for delimited-text files. When `sep=None`
(default) a `.tsv` file uses a tab and everything else uses a comma; `sep` is
ignored for Excel and Parquet.

```python
df = load_table("station.dat", sep=";")
```

## Saving: `save_table`

`save_table` writes CSV, Excel, or Parquet, again inferring the format from the
suffix, and returns the `Path` it wrote to. Missing parent directories are
created for you.

```python
from nextaire_tools import save_table

out = save_table(clean, "artifacts/clean.parquet")
print(out)   # artifacts/clean.parquet
```

The `index` argument controls whether the index is written. When `index=None`
(the default), the index is written **only** if it carries information — a
`DatetimeIndex` or a named index — so a throwaway `RangeIndex` is dropped:

```python
save_table(df, "clean.csv")               # DatetimeIndex -> written as a column
save_table(df, "clean.csv", index=False)  # force-drop the index
```

!!! warning "A CSV round-trip does not remember your datetime index"
    Writing a datetime-indexed frame to CSV stores the timestamps as plain text.
    When you read it back you must re-specify `time_col` to recover the index:

    ```python
    save_table(df, "clean.csv")                      # index -> "timestamp" text column
    again = load_table("clean.csv",
                       time_col="timestamp", set_time_index=True)
    ```

    Parquet preserves dtypes, so it survives a round-trip without re-parsing —
    prefer it for intermediate artifacts.

!!! example "Load, clean, and persist"
    ```python
    from nextaire_tools import load_table, save_table, Pipeline
    from nextaire_tools.preprocessing import MissingValueHandler, Scaler

    df = load_table("station.csv", time_col="timestamp", set_time_index=True)
    clean = Pipeline([
        MissingValueHandler(strategy="interpolate", limit=3),
        Scaler(method="standard"),
    ]).fit_transform(df)
    save_table(clean, "artifacts/clean.parquet")
    ```

## What `load_table` does *not* do

`load_table` is a tabular reader; it does not interpret domain-specific
encodings. In particular, **station coordinates stored as degrees–minutes–seconds
(DMS) strings** are a separate concern handled by the extractors module — see
[`load_stations`](../api/extractors.md) and the
[Copernicus extractors guide](extractors.md), which convert DMS to decimal
degrees for you.

## See also

- [Missing values](missing-values.md) — the natural next step after loading.
- [Pipelines](pipelines.md) — chain cleaning steps into one reusable object.
- [API reference: `nextaire_tools.io`](../api/io.md) — full signatures for `load_table`
  and `save_table`.
