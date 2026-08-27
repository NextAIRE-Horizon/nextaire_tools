"""Unit tests for extractors that do not require network access."""

from __future__ import annotations

import pandas as pd
import pytest

from nextaire_tools.extractors import (
    CAMSExtractor,
    ERA5Extractor,
    ERA5LandExtractor,
    dms_to_dd,
    load_stations,
    safe_filename,
)


def test_dms_to_dd_north_east():
    assert dms_to_dd("46°18'27\"N") == pytest.approx(46.3075, abs=1e-4)
    assert dms_to_dd("15°58'44\"E") == pytest.approx(15.97889, abs=1e-4)


def test_dms_to_dd_south_west_negative():
    assert dms_to_dd("10°00'00\"S") == pytest.approx(-10.0, abs=1e-6)
    assert dms_to_dd("20°30'00\"W") == pytest.approx(-20.5, abs=1e-6)


def test_dms_to_dd_passthrough_number():
    assert dms_to_dd(45.81) == pytest.approx(45.81)


def test_safe_filename():
    assert safe_filename("Zagreb / Centar #1") == "Zagreb_Centar_1" or " " not in safe_filename(
        "Zagreb / Centar #1"
    )


def test_load_stations_from_excel(tmp_path):
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "Measuring station ": ["Alpha", "Beta"],  # note trailing space in source
            "Longitude": ["15°58'44\"E", "16°26'24\"E"],
            "Latitude": ["45°48'36\"N", "46°18'27\"N"],
        }
    )
    path = tmp_path / "coords.xlsx"
    raw.to_excel(path, index=False)

    stations = load_stations(path)
    assert list(stations.columns) == ["station_name", "station_lon", "station_lat"]
    assert stations.loc[0, "station_name"] == "Alpha"
    assert stations["station_lon"].iloc[1] == pytest.approx(16.44, abs=1e-2)


def test_era5_build_request():
    ex = ERA5Extractor()
    req = ex.build_request(area=[49.1, 9.53, 46.3, 17.16], start="2024-01-01", end="2024-01-02")
    assert "2024" in req["year"]
    assert "01" in req["month"]
    assert set(req["day"]) >= {"01", "02"}
    assert len(req["time"]) == 24
    assert req["area"] == [49.1, 9.53, 46.3, 17.16]
    assert req["variable"] == ex.DEFAULT_VARIABLES


def test_cams_build_request_uses_date_range():
    ex = CAMSExtractor()
    req = ex.build_request(area=[49, 9, 46, 17], start="2024-01-01", end="2024-01-03")
    assert req["date"] == "2024-01-01/2024-01-03"


def test_land_request_has_no_product_type():
    ex = ERA5LandExtractor()
    req = ex.build_request(area=[49, 9, 46, 17], start="2024-01-01", end="2024-01-01")
    assert "product_type" not in req


def test_store_urls():
    assert ERA5Extractor.API_URL == "https://cds.climate.copernicus.eu/api"
    assert CAMSExtractor.API_URL == "https://ads.atmosphere.copernicus.eu/api"
    assert ERA5LandExtractor.DEFAULT_DATASET == "reanalysis-era5-land"
