"""Input-validation helpers shared across :mod:`nextaire_tools`.

These functions centralise the "coerce, check, and resolve columns" logic so
every preprocessing step behaves consistently and raises the same, informative
errors.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from importlib import import_module
from types import ModuleType

import pandas as pd
from pandas.api import types as pdt

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import (
    ColumnNotFoundError,
    MissingDependencyError,
    SchemaError,
)

__all__ = [
    "check_dataframe",
    "resolve_columns",
    "ensure_datetime_index",
    "require",
]


def check_dataframe(X: object, *, copy: bool = True) -> pd.DataFrame:
    """Validate that ``X`` is a non-empty :class:`pandas.DataFrame`.

    Parameters
    ----------
    X:
        Object to validate.
    copy:
        When ``True`` (default) a defensive copy is returned so that in-place
        mutation by a step never touches the caller's frame.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    SchemaError
        If ``X`` is not a DataFrame or has no columns.
    """
    if not isinstance(X, pd.DataFrame):
        raise SchemaError(
            f"Expected a pandas.DataFrame, got {type(X).__name__!r}. "
            "Use nextaire_tools.load_table(...) to read a file into a DataFrame first."
        )
    if X.shape[1] == 0:
        raise SchemaError("Input DataFrame has no columns.")
    return X.copy() if copy else X


def _as_label_list(columns: ColumnLike) -> list[Hashable]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Iterable):
        return [columns]  # single scalar label
    return list(columns)


def resolve_columns(
    df: pd.DataFrame,
    columns: ColumnLike | None = None,
    *,
    numeric_only: bool = False,
    exclude: Sequence[Hashable] | None = None,
) -> list[Hashable]:
    """Resolve a user column selection into a concrete, validated list.

    Parameters
    ----------
    df:
        The frame the columns must exist in.
    columns:
        Explicit selection. When ``None`` all columns are used (optionally
        filtered to numeric dtypes via ``numeric_only``).
    numeric_only:
        When ``columns`` is ``None``, restrict the default selection to numeric
        columns. Ignored when ``columns`` is given explicitly.
    exclude:
        Labels to drop from the resolved list (e.g. a target column).

    Returns
    -------
    list of hashable
        Column labels, order-preserving and de-duplicated.

    Raises
    ------
    ColumnNotFoundError
        If any explicitly requested column is missing.
    """
    exclude_set = set(exclude or ())

    if columns is None:
        if numeric_only:
            selected = [c for c in df.columns if pdt.is_numeric_dtype(df[c])]
        else:
            selected = list(df.columns)
    else:
        requested = _as_label_list(columns)
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ColumnNotFoundError(
                f"Columns not found in DataFrame: {missing}. Available columns: {list(df.columns)}"
            )
        selected = requested

    # Order-preserving de-duplication and exclusion.
    seen: set[Hashable] = set()
    out: list[Hashable] = []
    for c in selected:
        if c in exclude_set or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def ensure_datetime_index(
    df: pd.DataFrame,
    time_col: Hashable | None = None,
    *,
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Return ``df`` indexed by a :class:`~pandas.DatetimeIndex`.

    Resolution order:

    1. If ``time_col`` is given, that column is parsed with
       :func:`pandas.to_datetime` and set as the index.
    2. Otherwise, if the existing index is already datetime-like, it is used.
    3. Otherwise a :class:`SchemaError` is raised.

    Parameters
    ----------
    df:
        Input frame.
    time_col:
        Name of the column holding timestamps. ``None`` to use the existing
        index.
    copy:
        Return a copy rather than mutating ``df``.
    sort:
        Sort by the resulting index (ascending).

    Returns
    -------
    pandas.DataFrame
    """
    out = df.copy() if copy else df

    if time_col is not None:
        if time_col not in out.columns:
            raise ColumnNotFoundError(
                f"time_col={time_col!r} not found. Available: {list(out.columns)}"
            )
        idx = pd.to_datetime(out[time_col], errors="coerce", utc=False)
        if idx.isna().all():
            raise SchemaError(f"Column {time_col!r} could not be parsed as datetime.")
        out = out.drop(columns=[time_col])
        out.index = pd.DatetimeIndex(idx)
    elif isinstance(out.index, pd.DatetimeIndex):
        pass
    elif pdt.is_datetime64_any_dtype(out.index):
        out.index = pd.DatetimeIndex(out.index)
    elif pdt.is_object_dtype(out.index) or pdt.is_string_dtype(out.index):
        # Only string-like indices are safe to coerce. A numeric index (e.g. the
        # default RangeIndex from read_csv) would be silently misread as epoch
        # nanoseconds, so those are rejected in the ``else`` branch below.
        coerced = pd.to_datetime(out.index, errors="coerce")
        if isinstance(coerced, pd.DatetimeIndex) and not coerced.isna().all():
            out.index = coerced
        else:
            raise SchemaError(
                "No datetime index found. Pass time_col=<column name> or set a "
                "DatetimeIndex on the frame first."
            )
    else:
        raise SchemaError(
            f"Cannot use a {out.index.dtype} index as a datetime index (a numeric "
            "index would be misread as epoch nanoseconds). Pass time_col=<column "
            "name> or set a DatetimeIndex on the frame first."
        )

    if sort:
        out = out.sort_index()
    return out


def require(package: str, extra: str, feature: str = "") -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Parameters
    ----------
    package:
        Importable module name (e.g. ``"torch"``).
    extra:
        The pip extra that provides it (e.g. ``"deep"``).
    feature:
        Human-readable feature description used in the error message.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    MissingDependencyError
        If the import fails.
    """
    try:
        return import_module(package)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MissingDependencyError(package, extra, feature) from exc
