"""CAMS global reanalysis extractor (Atmosphere Data Store).

CAMS (Copernicus Atmosphere Monitoring Service) global reanalysis EAC4
provides three-hourly atmospheric-composition fields — particulate matter and
reactive gases — distributed through the Atmosphere Data Store (ADS). This
module provides :class:`CAMSExtractor`, targeting the
``cams-global-reanalysis-eac4`` dataset.

.. note::
   ADS uses a *different* endpoint and Personal Access Token from the CDS. Make
   sure your ``~/.cdsapirc`` (or ``key=`` / ``CDSAPI_KEY``) matches the ADS
   store; the default :attr:`CAMSExtractor.API_URL` points there.

Requesting CAMS **forecasts** instead of the reanalysis
-------------------------------------------------------
The near-real-time forecast dataset ``cams-global-atmospheric-composition-forecasts``
uses the same building blocks but additionally requires ``"leadtime_hour"`` (a
list of forecast lead times, e.g. ``["0", "12", "24"]``) and ``"type"`` (e.g.
``"forecast"``). Build such a request by extending the dictionary returned here,
for example::

    request = CAMSExtractor().build_request(area=..., start=..., end=...)
    request["leadtime_hour"] = ["0", "12", "24"]
    request["type"] = "forecast"
    extractor.retrieve(request, "cams_fc.zip",
                       dataset="cams-global-atmospheric-composition-forecasts")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.extractors.base import CopernicusExtractor
from nextaire_tools.utils.logging import get_logger

__all__ = ["CAMSExtractor"]

_LOG = get_logger(__name__)


class CAMSExtractor(CopernicusExtractor):
    """Extractor for CAMS global reanalysis (EAC4) atmospheric composition.

    Examples
    --------
    >>> extractor = CAMSExtractor()  # doctest: +SKIP
    >>> frames = extractor.extract_to_frames(  # doctest: +SKIP
    ...     stations=stations,
    ...     area=[49.1, 9.5, 46.3, 17.2],
    ...     start="2023-01-01",
    ...     end="2023-01-07",
    ... )
    """

    STORE_NAME = "Atmosphere Data Store (ADS)"
    API_URL = "https://ads.atmosphere.copernicus.eu/api"
    DEFAULT_DATASET = "cams-global-reanalysis-eac4"
    DEFAULT_VARIABLES: ClassVar[list[str]] = [
        "particulate_matter_2.5um",
        "particulate_matter_10um",
        "nitrogen_dioxide",
        "ozone",
        "carbon_monoxide",
        "sulphur_dioxide",
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
        """Build a CAMS EAC4 reanalysis request dictionary.

        Unlike ERA5, EAC4 uses a single ``date`` range string
        (``"YYYY-MM-DD/YYYY-MM-DD"``) rather than separate year/month/day lists.

        Parameters
        ----------
        variables:
            Variable names. Defaults to :attr:`DEFAULT_VARIABLES`.
        area:
            Bounding box ``[North, West, South, East]`` in decimal degrees.
        start, end:
            Inclusive date-range bounds (anything :class:`pandas.Timestamp`
            accepts). Combined into a single ``"start/end"`` date string.
        times:
            List of ``"HH:00"`` times. Defaults to EAC4's three-hourly steps
            (``00:00, 03:00, …, 21:00``).
        data_format:
            ``"grib"`` (default) or ``"netcdf"``. GRIB is required for the
            downstream GRIB sampling pipeline.
        download_format:
            ``"zip"`` (default) or ``"unarchived"``.

        Returns
        -------
        dict
            A request with keys ``variable``, ``date``, ``time``,
            ``data_format``, ``download_format`` and ``area``.

        Raises
        ------
        ConfigurationError
            If ``area`` is malformed or ``end`` precedes ``start``.
        """
        variables = list(variables) if variables is not None else list(self.DEFAULT_VARIABLES)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # Compare at day granularity to match the emitted %Y-%m-%d date range.
        if end_ts.normalize() < start_ts.normalize():
            raise ConfigurationError(f"end ({end}) is before start ({start}).")
        date_range = f"{start_ts:%Y-%m-%d}/{end_ts:%Y-%m-%d}"

        # EAC4 is available every three hours.
        time_list = list(times) if times is not None else self._hourly_times(3)
        box = self._normalize_area(area)

        return {
            "variable": variables,
            "date": date_range,
            "time": time_list,
            "data_format": data_format,
            "download_format": download_format,
            "area": box,
        }
