"""Station-metadata loading for Copernicus extractors.

This module turns a spreadsheet (or CSV/Parquet) of monitoring-station
coordinates into a tidy :class:`pandas.DataFrame` with the exact three columns
the sampling routines expect: ``station_name``, ``station_lon`` and
``station_lat`` (both in **decimal degrees**).

The original data sources frequently store coordinates as
degrees-minutes-seconds (DMS) strings such as ``"46°18'27\\"N"`` and sometimes
carry stray whitespace in their headers (e.g. a trailing space in
``"Measuring station "``). :func:`dms_to_dd` and :func:`load_stations` handle
both robustly.
"""

from __future__ import annotations

import re
from collections.abc import Hashable

import pandas as pd

from nextaire_tools._typing import PathType
from nextaire_tools.exceptions import SchemaError
from nextaire_tools.io.readers import load_table
from nextaire_tools.utils.logging import get_logger

__all__ = ["dms_to_dd", "load_stations"]

_LOG = get_logger(__name__)

# Matches unsigned integers / decimals inside a (cleaned) DMS string. The sign
# is handled separately (a leading minus or an S/W hemisphere marker), so that
# hyphen-delimited forms like "46-18-27N" treat the hyphens purely as separators.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def dms_to_dd(value: object) -> float | object:
    """Convert a degrees-minutes-seconds string to decimal degrees.

    Non-string inputs (values that are already numeric, ``NaN``, ``None``, …)
    are returned unchanged, so the function is safe to apply column-wide with
    :meth:`pandas.Series.map`.

    Parameters
    ----------
    value:
        A DMS string such as ``"46°18'27\\"N"`` or ``"9 31 48 E"``. The
        degree (``°``), minute (``'``) and second (``"``) markers are optional;
        any run of non-numeric characters is treated as a separator. A trailing
        ``S`` or ``W`` hemisphere marker (or a leading minus sign) yields a
        negative result.

    Returns
    -------
    float or object
        The coordinate in decimal degrees when ``value`` is a parseable string;
        otherwise ``value`` itself, unchanged.

    Examples
    --------
    >>> round(dms_to_dd("46°18'27\\"N"), 5)
    46.3075
    >>> round(dms_to_dd("9°31'48\\"W"), 5)
    -9.53
    >>> dms_to_dd(46.3075)
    46.3075
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return value

    degrees = float(numbers[0])
    minutes = float(numbers[1]) if len(numbers) > 1 else 0.0
    seconds = float(numbers[2]) if len(numbers) > 2 else 0.0

    magnitude = degrees + minutes / 60.0 + seconds / 3600.0

    upper = text.upper()
    negative = text.lstrip().startswith("-") or "S" in upper or "W" in upper
    return -magnitude if negative else magnitude


def _build_stripped_lookup(columns: pd.Index) -> dict[Hashable, Hashable]:
    """Map each column's whitespace-stripped label back to the real label."""
    lookup: dict[Hashable, Hashable] = {}
    for col in columns:
        key = col.strip() if isinstance(col, str) else col
        lookup.setdefault(key, col)
    return lookup


def load_stations(
    path: PathType,
    *,
    name_col: str = "Measuring station",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    index_col: int | str | None = 0,
) -> pd.DataFrame:
    """Load monitoring-station metadata into a tidy DataFrame.

    The file is read with :func:`nextaire_tools.io.load_table` (CSV / Excel / Parquet).
    Candidate column names are matched *after* stripping surrounding whitespace,
    so a header like ``"Measuring station "`` still resolves to ``name_col``.
    Longitude and latitude cells are passed through :func:`dms_to_dd`, so both
    decimal-degree and DMS-string source formats are accepted.

    Parameters
    ----------
    path:
        Path to the station file (``.xlsx``, ``.csv``, ``.parquet``, …).
    name_col:
        Header of the station-name column. Default ``"Measuring station"``.
    lon_col:
        Header of the longitude column. Default ``"Longitude"``.
    lat_col:
        Header of the latitude column. Default ``"Latitude"``.
    index_col:
        Column to use as the row index when reading (forwarded to the pandas
        reader). Default ``0`` matches the source spreadsheets whose first
        column is a numeric id. Pass ``None`` to disable.

    Returns
    -------
    pandas.DataFrame
        A frame with a default :class:`~pandas.RangeIndex` and exactly the
        columns ``["station_name", "station_lon", "station_lat"]``. Coordinates
        are floats in decimal degrees.

    Raises
    ------
    SchemaError
        If any of the requested columns cannot be found (the message lists the
        columns that are available).

    Examples
    --------
    >>> stations = load_stations("data/Coordinates.xlsx")  # doctest: +SKIP
    >>> list(stations.columns)  # doctest: +SKIP
    ['station_name', 'station_lon', 'station_lat']
    """
    df = load_table(path, index_col=index_col)

    lookup = _build_stripped_lookup(df.columns)

    def _resolve(target: str) -> Hashable | None:
        if target in df.columns:
            return target
        key = target.strip() if isinstance(target, str) else target
        return lookup.get(key)

    resolved: dict[str, Hashable] = {}
    missing: list[str] = []
    for out_label, target in (
        ("station_name", name_col),
        ("station_lon", lon_col),
        ("station_lat", lat_col),
    ):
        actual = _resolve(target)
        if actual is None:
            missing.append(target)
        else:
            resolved[out_label] = actual

    if missing:
        raise SchemaError(
            f"Station column(s) not found: {missing}. Available columns: {list(df.columns)}"
        )

    lon = df[resolved["station_lon"]].map(dms_to_dd)
    lat = df[resolved["station_lat"]].map(dms_to_dd)

    out = pd.DataFrame(
        {
            "station_name": df[resolved["station_name"]].to_numpy(),
            "station_lon": pd.to_numeric(lon, errors="coerce").to_numpy(),
            "station_lat": pd.to_numeric(lat, errors="coerce").to_numpy(),
        }
    )

    _LOG.info("Loaded %d station(s) from %s", len(out), path)
    return out
