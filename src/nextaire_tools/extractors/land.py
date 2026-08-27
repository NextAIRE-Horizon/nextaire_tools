"""ERA5-Land reanalysis extractor (Climate Data Store).

ERA5-Land is a replay of the ERA5 land component at higher (~9 km) resolution,
distributed hourly through the Copernicus Climate Data Store (CDS). This module
provides :class:`ERA5LandExtractor`, targeting the ``reanalysis-era5-land``
dataset with a set of land-surface variables useful as air-quality predictors.

The request is built like ERA5's but **without** a ``product_type`` key —
ERA5-Land has no product-type dimension.

Land-cover / vegetation raster products
----------------------------------------
Dedicated Copernicus Land Monitoring Service (CLMS) raster products — land
cover, leaf-area index (LAI), NDVI and similar — are **not** available through
``cdsapi``. They are distributed via WEkEO and its Harmonised Data Access (HDA)
API instead. For per-station *land-use* features (fraction of road / building /
green space around each site), the OpenStreetMap land-use-regression (LUR)
approach demonstrated in the project's ``notebooks/LurExtractor.ipynb`` (using
``osmnx``/``geopandas``, pip extra ``"geo"``) is a practical alternative that
does not require any Copernicus raster download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from nextaire_tools.extractors.base import CopernicusExtractor
from nextaire_tools.utils.logging import get_logger

__all__ = ["ERA5LandExtractor"]

_LOG = get_logger(__name__)


class ERA5LandExtractor(CopernicusExtractor):
    """Extractor for ERA5-Land hourly reanalysis.

    Examples
    --------
    >>> extractor = ERA5LandExtractor()  # doctest: +SKIP
    >>> frames = extractor.extract_to_frames(  # doctest: +SKIP
    ...     stations=stations,
    ...     area=[49.1, 9.5, 46.3, 17.2],
    ...     start="2026-01-01",
    ...     end="2026-01-31",
    ... )
    """

    STORE_NAME = "Climate Data Store (CDS) — ERA5-Land"
    API_URL = "https://cds.climate.copernicus.eu/api"
    DEFAULT_DATASET = "reanalysis-era5-land"
    DEFAULT_VARIABLES: ClassVar[list[str]] = [
        "2m_temperature",
        "total_precipitation",
        "volumetric_soil_water_layer_1",
        "leaf_area_index_high_vegetation",
        "skin_temperature",
        "snow_depth",
    ]

    def build_request(  # type: ignore[override]  # subclasses specialize the request builder
        self,
        *,
        variables: Sequence[str] | None = None,
        area: Sequence[float],
        start: str,
        end: str,
        times: Sequence[str] | None = None,
        data_format: str = "grib",
        download_format: str = "zip",
    ) -> dict[str, Any]:
        """Build an ERA5-Land request dictionary.

        Identical in shape to :meth:`ERA5Extractor.build_request` but with no
        ``product_type`` key, which ERA5-Land does not accept.

        Parameters
        ----------
        variables:
            Variable names. Defaults to :attr:`DEFAULT_VARIABLES`.
        area:
            Bounding box ``[North, West, South, East]`` in decimal degrees.
        start, end:
            Inclusive date-range bounds (anything :class:`pandas.Timestamp`
            accepts). Expanded into CDS ``year``/``month``/``day`` lists; see
            :meth:`~nextaire_tools.extractors.base.CopernicusExtractor._split_dates` for
            the Cartesian-grid caveat on multi-month ranges.
        times:
            List of ``"HH:00"`` times. Defaults to every hour (00:00..23:00).
        data_format:
            ``"grib"`` (default) or ``"netcdf"``. GRIB is required for the
            downstream GRIB sampling pipeline.
        download_format:
            ``"zip"`` (default) or ``"unarchived"``.

        Returns
        -------
        dict
            A request with keys ``variable``, ``year``, ``month``, ``day``,
            ``time``, ``data_format``, ``download_format`` and ``area``.

        Raises
        ------
        ConfigurationError
            If ``area`` is malformed or ``end`` precedes ``start``.
        """
        variables = list(variables) if variables is not None else list(self.DEFAULT_VARIABLES)
        years, months, days = self._split_dates(start, end)
        time_list = list(times) if times is not None else self._hourly_times(1)
        box = self._normalize_area(area)

        return {
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": time_list,
            "data_format": data_format,
            "download_format": download_format,
            "area": box,
        }
