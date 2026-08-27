"""Unified tabular IO for air-quality data.

A single entry point, :func:`load_table`, reads CSV, Excel, and Parquet files
into a :class:`pandas.DataFrame` with optional datetime-index handling, so the
rest of the pipeline never has to care about the on-disk format.
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd

from nextaire_tools._typing import PathType
from nextaire_tools.exceptions import SchemaError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import ensure_datetime_index

__all__ = ["load_table", "save_table", "SUPPORTED_SUFFIXES"]

_LOG = get_logger(__name__)

# Map file suffix -> pandas reader name. Values chosen so a single code path
# covers every supported format.
_CSV_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}
_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}
_PARQUET_SUFFIXES = {".parquet", ".pq", ".parq"}

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(_CSV_SUFFIXES | _EXCEL_SUFFIXES | _PARQUET_SUFFIXES)


def _infer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _EXCEL_SUFFIXES:
        return "excel"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    raise SchemaError(
        f"Unsupported file type {suffix!r} for {path.name!r}. "
        f"Supported extensions: {sorted(SUPPORTED_SUFFIXES)}."
    )


def load_table(
    path: PathType,
    *,
    time_col: Hashable | None = None,
    set_time_index: bool = False,
    sheet_name: str | int = 0,
    sep: str | None = None,
    columns: list[Hashable] | None = None,
    **reader_kwargs: Any,
) -> pd.DataFrame:
    """Load a tabular file (CSV / Excel / Parquet) into a DataFrame.

    The file format is inferred from the extension. Extra keyword arguments are
    forwarded to the underlying pandas reader (``read_csv`` / ``read_excel`` /
    ``read_parquet``), so anything pandas supports is available.

    Parameters
    ----------
    path:
        Path to the data file.
    time_col:
        Name of a column holding timestamps. When given it is parsed with
        :func:`pandas.to_datetime`.
    set_time_index:
        When ``True`` (and ``time_col`` resolves to a datetime), the frame is
        returned indexed by a sorted :class:`~pandas.DatetimeIndex`.
    sheet_name:
        Worksheet to read for Excel files. Ignored for other formats.
    sep:
        Field delimiter for delimited-text files. When ``None`` a ``.tsv`` file
        defaults to a tab and everything else to a comma. Ignored for
        Excel/Parquet.
    columns:
        Subset of columns to load (pushed down to Parquet readers for
        efficiency; applied post-read for other formats).
    **reader_kwargs:
        Passed through to the pandas reader.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SchemaError
        If the extension is not supported.

    Examples
    --------
    >>> df = load_table("station.csv", time_col="timestamp", set_time_index=True)
    >>> df = load_table("obs.parquet", columns=["timestamp", "no2", "o3"])
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    kind = _infer_kind(p)
    _LOG.debug("Loading %s as %s", p, kind)

    # When a column subset is requested, always read time_col too (it is needed
    # for parsing / indexing) even if the caller left it out of ``columns``.
    read_columns: list[Hashable] | None = list(columns) if columns is not None else None
    if read_columns is not None and time_col is not None and time_col not in read_columns:
        read_columns = [*read_columns, time_col]

    if kind == "csv":
        if sep is None:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, **reader_kwargs)
    elif kind == "excel":
        df = pd.read_excel(p, sheet_name=sheet_name, **reader_kwargs)
        if isinstance(df, dict):  # multiple sheets requested
            raise SchemaError(
                "read_excel returned multiple sheets; pass a single sheet_name "
                "(str or int) to load_table."
            )
    else:  # parquet
        if read_columns is not None:
            reader_kwargs.setdefault("columns", read_columns)
        df = pd.read_parquet(p, **reader_kwargs)

    if read_columns is not None and kind != "parquet":
        missing = [c for c in read_columns if c not in df.columns]
        if missing:
            raise SchemaError(f"Requested columns not present: {missing}")
        df = df[read_columns]

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if set_time_index:
            df = ensure_datetime_index(df, time_col=time_col, copy=False)
    elif set_time_index:
        df = ensure_datetime_index(df, copy=False)

    _LOG.info("Loaded %s: %d rows x %d columns", p.name, len(df), df.shape[1])
    return df


def save_table(
    df: pd.DataFrame, path: PathType, *, index: bool | None = None, **kwargs: Any
) -> Path:
    """Write a DataFrame to CSV / Excel / Parquet, inferring format from suffix.

    Parameters
    ----------
    df:
        Frame to write.
    path:
        Destination path. Parent directories are created if needed.
    index:
        Whether to write the index. When ``None`` the index is written unless it
        is a plain, unnamed ``RangeIndex`` — so datetime, named, and MultiIndex
        indices are preserved while a default row counter is dropped.
    **kwargs:
        Forwarded to the pandas writer.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    p = Path(path)
    kind = _infer_kind(p)  # validate the suffix before touching the filesystem
    p.parent.mkdir(parents=True, exist_ok=True)

    if index is None:
        index = (
            isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex))
            or df.index.name is not None
            or any(name is not None for name in df.index.names)
        )

    if kind == "csv":
        df.to_csv(p, index=index, **kwargs)
    elif kind == "excel":
        df.to_excel(p, index=index, **kwargs)
    else:  # parquet ignores the ``index`` semantics of csv; pandas handles it
        df.to_parquet(p, index=index, **kwargs)

    _LOG.info("Wrote %s (%d rows)", p, len(df))
    return p
