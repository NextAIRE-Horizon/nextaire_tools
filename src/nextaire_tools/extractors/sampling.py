"""GRIB archive handling and nearest-neighbour station sampling.

These helpers take the ZIP archive that a Copernicus data store returns,
extract the GRIB messages inside it, open them with ``cfgrib`` and sample every
field at each monitoring station's coordinates. The result is one tidy
:class:`pandas.DataFrame` per station, indexed by timestamp.

Optional dependencies
----------------------
``xarray`` and ``cfgrib`` (pip extra ``"extract"``) are imported lazily inside
:func:`open_grib_datasets` and :func:`sample_at_points` via
:func:`nextaire_tools.utils.validation.require`; the module itself imports cleanly with
only the core dependencies installed. :func:`extract_archive`,
:func:`merge_station_frames` and :func:`safe_filename` need neither and work
everywhere.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from nextaire_tools._typing import PathType
from nextaire_tools.exceptions import ExtractionError, SchemaError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import require

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

__all__ = [
    "extract_archive",
    "merge_station_frames",
    "open_grib_datasets",
    "safe_filename",
    "sample_at_points",
]

_LOG = get_logger(__name__)

# Recognised GRIB file suffixes (lower-cased for comparison).
_GRIB_SUFFIXES = {".grib", ".grb", ".grib2"}


def _looks_like_grib(path: Path) -> bool:
    """Return ``True`` if the file begins with the GRIB magic bytes (``b"GRIB"``)."""
    try:
        with path.open("rb") as fh:
            return fh.read(4) == b"GRIB"
    except OSError:  # pragma: no cover - unreadable file handled by caller
        return False


# Columns a ``stations`` frame must provide (see :func:`load_stations`).
_STATION_COLUMNS = ("station_name", "station_lon", "station_lat")


def extract_archive(zip_path: PathType, dest: PathType) -> list[Path]:
    """Extract a downloaded archive and return the GRIB files inside it.

    Parameters
    ----------
    zip_path:
        Path to the ``.zip`` archive returned by the data store. If the path is
        not a ZIP but is itself a GRIB file (some single-file requests download
        a raw GRIB), it is returned as-is.
    dest:
        Directory to extract into. Created if it does not exist.

    Returns
    -------
    list of pathlib.Path
        Sorted paths of every extracted file whose suffix is ``.grib``,
        ``.grb`` or ``.grib2``.

    Raises
    ------
    ExtractionError
        If the archive is missing, is not a valid ZIP, or contains no GRIB
        files.
    """
    zip_p = Path(zip_path)
    dest_p = Path(dest)

    if not zip_p.exists():
        raise ExtractionError(f"Archive not found: {zip_p}")

    if not zipfile.is_zipfile(zip_p):
        # Some single-file / download_format="unarchived" requests return a raw
        # GRIB. Detect it by suffix OR by its magic bytes, so the archive's name
        # (which the caller may not control) is irrelevant.
        if zip_p.suffix.lower() in _GRIB_SUFFIXES or _looks_like_grib(zip_p):
            _LOG.info("Input %s is a raw GRIB file (not a ZIP); using directly", zip_p.name)
            return [zip_p]
        raise ExtractionError(f"{zip_p} is neither a valid ZIP archive nor a GRIB file.")

    dest_p.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_p) as zf:
            zf.extractall(dest_p)
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Could not read ZIP archive {zip_p}: {exc}") from exc

    grib_files = sorted(p for p in dest_p.rglob("*") if p.suffix.lower() in _GRIB_SUFFIXES)
    if not grib_files:
        raise ExtractionError(f"No GRIB files (.grib/.grb/.grib2) found inside {zip_p}.")

    _LOG.info("Extracted %d GRIB file(s) from %s", len(grib_files), zip_p.name)
    return grib_files


def open_grib_datasets(grib_files: Sequence[PathType]) -> list[xr.Dataset]:
    """Open GRIB files as a flat list of :class:`xarray.Dataset` objects.

    ``cfgrib`` returns *one* dataset per ``typeOfLevel`` grouping within a GRIB
    file, so a single file may yield several datasets; they are all flattened
    into the returned list. ``indexpath=""`` disables cfgrib's on-disk index so
    read-only / temporary directories work.

    Parameters
    ----------
    grib_files:
        Paths to GRIB files (typically from :func:`extract_archive`).

    Returns
    -------
    list of xarray.Dataset

    Raises
    ------
    nextaire_tools.exceptions.MissingDependencyError
        If ``cfgrib`` (pip extra ``"extract"``) is not installed.
    ExtractionError
        If a GRIB file cannot be opened.
    """
    cfgrib = require("cfgrib", "extract", "reading GRIB reanalysis files")

    datasets: list[Any] = []
    for path in grib_files:
        try:
            datasets.extend(cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}))
        except Exception as exc:
            raise ExtractionError(f"Failed to open GRIB file {path}: {exc}") from exc

    _LOG.info("Opened %d GRIB dataset group(s)", len(datasets))
    return datasets


def sample_at_points(ds: xr.Dataset, stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sample every field in ``ds`` at each station via nearest neighbour.

    The dataset's horizontal coordinates are detected automatically
    (``longitude``/``lon`` and ``latitude``/``lat``) and its time coordinate is
    normalised to ``valid_time`` (falling back to ``time``) and renamed to
    ``timestamp``.

    Parameters
    ----------
    ds:
        A dataset opened from a GRIB file.
    stations:
        Station metadata with the columns ``station_name``, ``station_lon`` and
        ``station_lat`` (see :func:`load_stations`).

    Returns
    -------
    dict of {str: pandas.DataFrame}
        One frame per station, keyed by ``station_name``, indexed by a sorted
        ``timestamp`` index, with one column per data variable in ``ds``.

    Raises
    ------
    SchemaError
        If ``stations`` is missing a required column.
    nextaire_tools.exceptions.MissingDependencyError
        If ``xarray`` (pip extra ``"extract"``) is not installed.
    """
    xr_mod = require("xarray", "extract", "sampling GRIB datasets at station points")

    missing = [c for c in _STATION_COLUMNS if c not in stations.columns]
    if missing:
        raise SchemaError(
            f"`stations` frame is missing column(s) {missing}. "
            f"Available columns: {list(stations.columns)}"
        )

    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"

    point = ds.sel(
        {
            lon_name: xr_mod.DataArray(stations["station_lon"].to_numpy(), dims="station"),
            lat_name: xr_mod.DataArray(stations["station_lat"].to_numpy(), dims="station"),
        },
        method="nearest",
    )

    data_vars = list(ds.data_vars)
    df = point[data_vars].to_dataframe().reset_index()

    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "timestamp"})
    df["station_name"] = df["station"].map(dict(enumerate(stations["station_name"])))

    df = df[["timestamp", "station_name", *data_vars]]
    return {
        name: group.drop(columns="station_name").set_index("timestamp").sort_index()
        for name, group in df.groupby("station_name")
    }


def merge_station_frames(
    per_dataset: Sequence[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Concatenate per-station frames coming from several GRIB datasets.

    Each element of ``per_dataset`` is a ``{station_name: DataFrame}`` mapping
    (the output of :func:`sample_at_points`). For every station the frames are
    concatenated column-wise; duplicate columns (a variable appearing in more
    than one dataset group) are dropped, keeping the first occurrence, and the
    result is sorted by its ``timestamp`` index.

    Parameters
    ----------
    per_dataset:
        One ``{station_name: DataFrame}`` mapping per opened dataset.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        One merged frame per station, keyed by ``station_name``.
    """
    names: set[str] = set()
    for mapping in per_dataset:
        names.update(mapping.keys())

    merged: dict[str, pd.DataFrame] = {}
    for name in sorted(names):
        frames = [mapping[name] for mapping in per_dataset if name in mapping]
        joined = pd.concat(frames, axis=1)
        joined = joined.loc[:, ~joined.columns.duplicated()]
        joined.index.name = "timestamp"
        merged[name] = joined.sort_index()
    return merged


def safe_filename(name: str) -> str:
    """Return a filesystem-safe version of ``name``.

    Any run of characters outside ``[A-Za-z0-9._-]`` is collapsed to a single
    underscore and leading/trailing underscores are stripped.

    Parameters
    ----------
    name:
        Arbitrary label (typically a station name).

    Returns
    -------
    str
        A sanitised token safe to use as a file name stem.

    Examples
    --------
    >>> safe_filename("Zagreb-1 (centar)")
    'Zagreb-1_centar'
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
