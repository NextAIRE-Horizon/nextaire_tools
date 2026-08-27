"""ERA5 single-level reanalysis extractor (Climate Data Store).

ERA5 is ECMWF's global atmospheric reanalysis, distributed hourly on a regular
lat/lon grid through the Copernicus Climate Data Store (CDS). This module
provides :class:`ERA5Extractor`, a thin :class:`~nextaire_tools.extractors.base.CopernicusExtractor`
subclass targeting the ``reanalysis-era5-single-levels`` dataset with a set of
meteorological variables commonly used as air-quality model predictors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from nextaire_tools.extractors.base import CopernicusExtractor
from nextaire_tools.utils.logging import get_logger

__all__ = ["ERA5Extractor"]

_LOG = get_logger(__name__)


class ERA5Extractor(CopernicusExtractor):
    """Extractor for ERA5 single-level hourly reanalysis.

    Examples
    --------
    >>> extractor = ERA5Extractor()  # doctest: +SKIP
    >>> frames = extractor.extract_to_frames(  # doctest: +SKIP
    ...     stations=stations,
    ...     area=[49.1, 9.5, 46.3, 17.2],
    ...     start="2026-01-01",
    ...     end="2026-01-31",
    ...     save_dir="data/era5",
    ... )
    """

    STORE_NAME = "Climate Data Store (CDS)"
    API_URL = "https://cds.climate.copernicus.eu/api"
    DEFAULT_DATASET = "reanalysis-era5-single-levels"
    DEFAULT_VARIABLES: ClassVar[list[str]] = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure",
        "total_precipitation",
        "boundary_layer_height",
        "surface_net_solar_radiation",
    ]

    def build_request(  # type: ignore[override]  # subclasses specialize the request builder
        self,
        *,
        variables: Sequence[str] | None = None,
        area: Sequence[float],
        start: str,
        end: str,
        times: Sequence[str] | None = None,
        product_type: str = "reanalysis",
        data_format: str = "grib",
        download_format: str = "zip",
    ) -> dict[str, Any]:
        """Build an ERA5 single-level request dictionary.

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
        product_type:
            ERA5 product type. Default ``"reanalysis"``.
        data_format:
            ``"grib"`` (default) or ``"netcdf"``. GRIB is required for the
            downstream GRIB sampling pipeline.
        download_format:
            ``"zip"`` (default) or ``"unarchived"``.

        Returns
        -------
        dict
            A request with keys ``product_type``, ``variable``, ``year``,
            ``month``, ``day``, ``time``, ``data_format``, ``download_format``
            and ``area``.

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
            "product_type": product_type,
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": time_list,
            "data_format": data_format,
            "download_format": download_format,
            "area": box,
        }
