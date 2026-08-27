"""Copernicus data-store extractors (ERA5, CAMS, ERA5-Land) and station helpers.

Requires the ``extract`` optional dependencies (``pip install 'nextaire_tools[extract]'``)
to actually download and read data; the module itself imports without them.
"""

from __future__ import annotations

from nextaire_tools.extractors.base import CopernicusExtractor
from nextaire_tools.extractors.cams import CAMSExtractor
from nextaire_tools.extractors.era5 import ERA5Extractor
from nextaire_tools.extractors.land import ERA5LandExtractor
from nextaire_tools.extractors.sampling import (
    extract_archive,
    merge_station_frames,
    open_grib_datasets,
    safe_filename,
    sample_at_points,
)
from nextaire_tools.extractors.stations import dms_to_dd, load_stations

__all__ = [
    "CopernicusExtractor",
    "ERA5Extractor",
    "CAMSExtractor",
    "ERA5LandExtractor",
    "load_stations",
    "dms_to_dd",
    "extract_archive",
    "open_grib_datasets",
    "sample_at_points",
    "merge_station_frames",
    "safe_filename",
]
