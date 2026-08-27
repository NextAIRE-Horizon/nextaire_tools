"""Base class for Copernicus (CDS / ADS) reanalysis extractors.

:class:`CopernicusExtractor` wraps the ``cdsapi`` client with a consistent,
well-typed interface and an end-to-end :meth:`~CopernicusExtractor.extract_to_frames`
helper that downloads a request, extracts the GRIB archive, samples every field
at a set of monitoring stations and returns one tidy
:class:`pandas.DataFrame` per station.

Subclasses (:class:`~nextaire_tools.extractors.era5.ERA5Extractor`,
:class:`~nextaire_tools.extractors.cams.CAMSExtractor`,
:class:`~nextaire_tools.extractors.land.ERA5LandExtractor`) only override the class-level
metadata (``STORE_NAME``, ``API_URL``, ``DEFAULT_DATASET``,
``DEFAULT_VARIABLES``) and :meth:`~CopernicusExtractor.build_request`.

Optional dependencies
----------------------
``cdsapi`` (pip extra ``"extract"``) is imported lazily inside
:meth:`~CopernicusExtractor._client`, so the module imports cleanly with only
the core dependencies installed.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from nextaire_tools._typing import PathType
from nextaire_tools.exceptions import (
    ConfigurationError,
    CredentialsError,
    ExtractionError,
    SchemaError,
)
from nextaire_tools.extractors.sampling import (
    extract_archive,
    merge_station_frames,
    open_grib_datasets,
    safe_filename,
    sample_at_points,
)
from nextaire_tools.io.readers import save_table
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import require

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

__all__ = ["CopernicusExtractor"]

_LOG = get_logger(__name__)


class CopernicusExtractor:
    """Download and post-process Copernicus reanalysis data.

    This class is not used directly; instantiate one of its subclasses. It
    centralises client construction, the ``retrieve``-then-``download`` call and
    the GRIB -> per-station DataFrame pipeline.

    Parameters
    ----------
    url:
        Data-store API endpoint. Defaults to the subclass' :attr:`API_URL`.
    key:
        Personal Access Token. When ``None`` the ``CDSAPI_KEY`` environment
        variable is consulted, and failing that the token in ``~/.cdsapirc`` is
        used by ``cdsapi`` itself.
    output_dir:
        Default directory for saved outputs. Default ``"data"``.
    client:
        A pre-built client object (mainly for testing). When given it is used
        verbatim and no ``cdsapi`` import happens.
    quiet:
        Forwarded to ``cdsapi.Client(quiet=...)`` to silence its progress
        logging. Default ``True``.

    Attributes
    ----------
    STORE_NAME : str
        Human-readable name of the data store.
    API_URL : str
        Default API endpoint.
    DEFAULT_DATASET : str
        Dataset id used when ``dataset`` is not supplied.
    DEFAULT_VARIABLES : list of str
        Variables requested by :meth:`build_request` when none are given.
    """

    STORE_NAME: ClassVar[str] = "Copernicus Data Store"
    API_URL: ClassVar[str] = "https://cds.climate.copernicus.eu/api"
    DEFAULT_DATASET: ClassVar[str] = ""
    DEFAULT_VARIABLES: ClassVar[list[str]] = []

    def __init__(
        self,
        *,
        url: str | None = None,
        key: str | None = None,
        output_dir: PathType = "data",
        client: Any | None = None,
        quiet: bool = True,
    ) -> None:
        self.url: str | None = url if url is not None else self.API_URL
        self.key: str | None = key if key is not None else os.getenv("CDSAPI_KEY")
        self.output_dir: Path = Path(output_dir)
        self.quiet: bool = quiet
        self._client_obj: Any | None = client
        self._client_cache: Any | None = None

    # ------------------------------------------------------------------ client
    def _client(self) -> Any:
        """Return (and cache) the underlying ``cdsapi`` client.

        Returns
        -------
        Any
            The client object.

        Raises
        ------
        nextaire_tools.exceptions.MissingDependencyError
            If ``cdsapi`` (pip extra ``"extract"``) is not installed.
        CredentialsError
            If the client cannot be constructed (missing / invalid
            credentials).
        """
        if self._client_cache is not None:
            return self._client_cache
        if self._client_obj is not None:
            self._client_cache = self._client_obj
            return self._client_cache

        cdsapi = require("cdsapi", "extract", "Copernicus downloads")

        # Pass url/key only when set so that credentials configured in
        # ~/.cdsapirc are honoured for whatever is left unspecified.
        kwargs: dict[str, Any] = {"quiet": self.quiet}
        if self.url is not None:
            kwargs["url"] = self.url
        if self.key is not None:
            kwargs["key"] = self.key

        try:
            client = cdsapi.Client(**kwargs)
        except Exception as exc:
            raise CredentialsError(
                f"Could not initialise the {self.STORE_NAME} client. Supply a "
                "Personal Access Token via the `key=` argument, the CDSAPI_KEY "
                "environment variable, or a ~/.cdsapirc file containing:\n"
                f"    url: {self.url}\n"
                "    key: <your Personal Access Token>\n"
                "See https://cds.climate.copernicus.eu/how-to-api for setup "
                f"instructions. Original error: {exc}"
            ) from exc

        self._client_cache = client
        return client

    # --------------------------------------------------------------- retrieval
    def retrieve(
        self,
        request: dict[str, Any],
        target: PathType,
        dataset: str | None = None,
    ) -> Path:
        """Submit a request and download the result to ``target``.

        Parameters
        ----------
        request:
            The data-store request dictionary (see :meth:`build_request`).
        target:
            Destination file path. Parent directories are created if needed.
        dataset:
            Dataset id. Defaults to :attr:`DEFAULT_DATASET`.

        Returns
        -------
        pathlib.Path
            The path the archive was downloaded to.

        Raises
        ------
        ConfigurationError
            If no dataset is available.
        ExtractionError
            If the download fails for any reason.
        """
        ds = dataset or self.DEFAULT_DATASET
        if not ds:
            raise ConfigurationError(
                "No dataset specified and DEFAULT_DATASET is empty; pass dataset=... explicitly."
            )

        target_p = Path(target)
        target_p.parent.mkdir(parents=True, exist_ok=True)

        client = self._client()
        _LOG.info("Requesting %r from %s -> %s", ds, self.STORE_NAME, target_p)
        try:
            client.retrieve(ds, request).download(str(target_p))
        except Exception as exc:
            raise ExtractionError(
                f"Failed to retrieve dataset {ds!r} from {self.STORE_NAME}: {exc}"
            ) from exc

        return target_p

    # ------------------------------------------------------------ request hook
    def build_request(self, **kwargs: Any) -> dict[str, Any]:
        """Build a data-store request dictionary.

        Subclasses must implement this.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement build_request().")

    # ------------------------------------------------------------- end-to-end
    def extract_to_frames(
        self,
        request: dict[str, Any] | None = None,
        *,
        stations: pd.DataFrame,
        dataset: str | None = None,
        save_dir: PathType | None = None,
        **build_kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Download, sample at stations and return one DataFrame per station.

        The full pipeline is: build the request (if not supplied) via
        :meth:`build_request`, download it to a temporary directory, extract the
        GRIB archive, open every dataset group, sample each at the station
        coordinates and merge the per-station frames. Optionally the result is
        written to CSV.

        Parameters
        ----------
        request:
            An explicit request dictionary. When ``None`` it is built from
            ``**build_kwargs`` via :meth:`build_request`.
        stations:
            Station metadata with ``station_name``, ``station_lon`` and
            ``station_lat`` columns (see :func:`~nextaire_tools.extractors.stations.load_stations`).
        dataset:
            Dataset id. Defaults to :attr:`DEFAULT_DATASET`.
        save_dir:
            When given, each station frame is written there as
            ``<safe_station_name>.csv``.
        **build_kwargs:
            Forwarded to :meth:`build_request` when ``request`` is ``None``.

        Returns
        -------
        dict of {str: pandas.DataFrame}
            One timestamp-indexed frame per station.

        Raises
        ------
        SchemaError
            If ``stations`` is not a DataFrame.
        ConfigurationError
            If both ``request`` and ``build_kwargs`` are supplied.
        ExtractionError
            If the download or GRIB extraction fails.
        """
        if not isinstance(stations, pd.DataFrame):
            raise SchemaError(
                "`stations` must be a pandas.DataFrame; use "
                "nextaire_tools.extractors.stations.load_stations(...) to build one."
            )

        if request is None:
            request = self.build_request(**build_kwargs)
        elif build_kwargs:
            raise ConfigurationError(
                "Pass either an explicit `request` or build_* keyword arguments, not both."
            )

        ds = dataset or self.DEFAULT_DATASET

        with tempfile.TemporaryDirectory(prefix="nextaire_tools_copernicus_") as tmp_str:
            tmp = Path(tmp_str)
            # Name the download to match its format so extract_archive can route
            # it (a raw GRIB is also detected by magic bytes as a backstop).
            download_format = str(request.get("download_format", "zip"))
            if download_format == "zip":
                archive = tmp / "download.zip"
            else:
                data_format = str(request.get("data_format", "grib"))
                suffix = ".nc" if data_format.startswith("netcdf") else ".grib"
                archive = tmp / f"download{suffix}"
            self.retrieve(request, archive, dataset=ds)

            grib_files = extract_archive(archive, tmp / "extracted")
            datasets = open_grib_datasets(grib_files)
            per_dataset = [sample_at_points(d, stations) for d in datasets]
            merged = merge_station_frames(per_dataset)

        _LOG.info("Built frames for %d station(s)", len(merged))

        if save_dir is not None:
            save_path = Path(save_dir)
            for name, frame in merged.items():
                save_table(frame, save_path / f"{safe_filename(name)}.csv", index=True)
            _LOG.info("Saved %d station CSV(s) to %s", len(merged), save_path)

        return merged

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _split_dates(start: str, end: str) -> tuple[list[str], list[str], list[str]]:
        """Split an inclusive date range into CDS year/month/day lists.

        The Climate Data Store treats ``year``, ``month`` and ``day`` as
        independent lists and retrieves their Cartesian product, silently
        ignoring dates that do not exist. This mirrors that convention.

        .. note::
           For ranges that span several months or years the returned lists form
           a *grid*: e.g. ``2020-12-30`` to ``2021-01-02`` yields years
           ``["2020", "2021"]`` and months ``["01", "12"]``, so the request also
           covers January 2020 and December 2021. Slice such ranges per-month if
           you need an exact window.

        Parameters
        ----------
        start, end:
            Inclusive range bounds; anything :class:`pandas.Timestamp` accepts
            (e.g. ``"2026-01-01"``).

        Returns
        -------
        tuple of (list of str, list of str, list of str)
            Zero-padded, sorted ``(years, months, days)``.

        Raises
        ------
        ConfigurationError
            If ``end`` precedes ``start``.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # Compare at day granularity — the grid below is built on normalized
        # dates, so a same-day range with a reversed clock time is still valid.
        if end_ts.normalize() < start_ts.normalize():
            raise ConfigurationError(f"end ({end}) is before start ({start}).")

        rng = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
        years = sorted({f"{d.year:04d}" for d in rng})
        months = sorted({f"{d.month:02d}" for d in rng})
        days = sorted({f"{d.day:02d}" for d in rng})
        return years, months, days

    @staticmethod
    def _hourly_times(step: int = 1) -> list[str]:
        """Return ``"HH:00"`` strings from 00:00 up to 23:00 at ``step`` hours."""
        if step < 1 or step > 24:
            raise ConfigurationError(f"time step must be in 1..24, got {step}.")
        return [f"{h:02d}:00" for h in range(0, 24, step)]

    @staticmethod
    def _normalize_area(area: Sequence[float] | None) -> list[float]:
        """Validate and coerce a CDS ``area`` box to ``[North, West, South, East]``."""
        if area is None:
            raise ConfigurationError(
                "`area` is required: pass [North, West, South, East] in degrees."
            )
        box = [float(v) for v in area]
        if len(box) != 4:
            raise ConfigurationError(f"`area` must have 4 values [N, W, S, E]; got {list(area)!r}.")
        return box

    @staticmethod
    def expand_area(stations: pd.DataFrame, *, margin: float = 0.5) -> list[float]:
        """Compute a CDS ``area`` box covering all stations plus a margin.

        Parameters
        ----------
        stations:
            Frame with ``station_lat`` and ``station_lon`` columns.
        margin:
            Degrees of padding added on every side. Default ``0.5``.

        Returns
        -------
        list of float
            ``[North, West, South, East]`` suitable for :meth:`build_request`.

        Raises
        ------
        SchemaError
            If the required coordinate columns are missing.
        """
        required: Iterable[str] = ("station_lat", "station_lon")
        missing = [c for c in required if c not in stations.columns]
        if missing:
            raise SchemaError(
                f"`stations` frame is missing column(s) {missing} needed to "
                f"compute an area box. Available: {list(stations.columns)}"
            )

        lat = pd.to_numeric(stations["station_lat"], errors="coerce")
        lon = pd.to_numeric(stations["station_lon"], errors="coerce")
        north = float(lat.max()) + margin
        south = float(lat.min()) - margin
        west = float(lon.min()) - margin
        east = float(lon.max()) + margin
        return [north, west, south, east]
