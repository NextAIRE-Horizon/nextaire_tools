"""Regression tests for bugs found and fixed during adversarial code review.

Each test pins one confirmed defect so it cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.exceptions import SchemaError
from nextaire_tools.extractors import CAMSExtractor, ERA5Extractor, dms_to_dd, extract_archive
from nextaire_tools.io import load_table, save_table
from nextaire_tools.models import BlockingTimeSeriesSplit, cross_val_report, make_regressor
from nextaire_tools.preprocessing import MissingValueHandler
from nextaire_tools.utils.validation import ensure_datetime_index


# 1 (HIGH) — ensure_datetime_index must NOT coerce a numeric index to epoch-ns.
def test_ensure_datetime_index_rejects_numeric_index():
    df = pd.DataFrame({"a": [1, 2, 3]})  # default RangeIndex
    with pytest.raises(SchemaError):
        ensure_datetime_index(df)


def test_load_table_set_time_index_without_time_col_rejects_numeric(tmp_path):
    p = tmp_path / "no_time.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(p, index=False)
    with pytest.raises(SchemaError):
        load_table(p, set_time_index=True)


# 2 (HIGH) — save_table must preserve a MultiIndex.
def test_save_table_preserves_multiindex(tmp_path):
    df = pd.DataFrame(
        {"val": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["station", "day"]
        ),
    )
    out = tmp_path / "multi.csv"
    save_table(df, out)
    reloaded = pd.read_csv(out)
    assert {"station", "day"}.issubset(reloaded.columns)


# 3 (MEDIUM) — cross_val_report must not exhaust a one-shot metrics iterable.
def test_cross_val_report_with_generator_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.1, 120)
    report = cross_val_report(
        make_regressor("linear"),
        X,
        y,
        cv=BlockingTimeSeriesSplit(n_splits=4),
        metrics=(m for m in ["mae", "rmse"]),  # generator
    )
    fold_rows = report.drop(index=["mean", "std"])
    assert not fold_rows.isna().to_numpy().any()  # every fold populated


# 4 (MEDIUM) — dms_to_dd must treat hyphens as separators, not minus signs.
def test_dms_to_dd_hyphen_separated():
    assert dms_to_dd("46-18-27N") == pytest.approx(46.3075, abs=1e-4)
    assert dms_to_dd("9-31-48-W") == pytest.approx(-9.53, abs=1e-4)


# 5 (MEDIUM) — extract_archive must detect a raw GRIB by magic bytes even if
#              the file is named .zip (download_format="unarchived").
def test_extract_archive_detects_grib_magic_bytes(tmp_path):
    fake_grib = tmp_path / "download.zip"  # misleading suffix, GRIB content
    fake_grib.write_bytes(b"GRIB\x00\x00\x00\x02rest-of-message")
    result = extract_archive(fake_grib, tmp_path / "out")
    assert result == [fake_grib]


# 6 (LOW) — same-calendar-day range with reversed clock times must be accepted.
def test_build_request_same_day_reversed_clock():
    req = ERA5Extractor().build_request(
        area=[49, 9, 46, 17], start="2024-03-10 10:00", end="2024-03-10 08:00"
    )
    assert req["day"] == ["10"]
    cams = CAMSExtractor().build_request(
        area=[49, 9, 46, 17], start="2024-03-10 10:00", end="2024-03-10 08:00"
    )
    assert cams["date"] == "2024-03-10/2024-03-10"


# 7 (LOW) — save_table must validate the suffix BEFORE creating directories.
def test_save_table_validates_suffix_before_mkdir(tmp_path):
    target = tmp_path / "new_dir" / "out.unsupported"
    with pytest.raises(SchemaError):
        save_table(pd.DataFrame({"a": [1]}), target)
    assert not target.parent.exists()  # no stray directory left behind


# 8 (LOW) — load_table(columns=...) that omits time_col must still work.
def test_load_table_columns_without_time_col(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="h", name="timestamp")
    pd.DataFrame({"no2": range(5), "o3": range(5)}, index=idx).to_csv(tmp_path / "s.csv")
    df = load_table(tmp_path / "s.csv", columns=["no2"], time_col="timestamp", set_time_index=True)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "no2" in df.columns


# 9 (LOW) — add_indicator must emit an indicator for threshold-dropped columns.
def test_add_indicator_covers_threshold_dropped_columns():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [np.nan, np.nan, np.nan, 4.0, 5.0],  # 60% missing -> dropped at 0.5
        }
    )
    step = MissingValueHandler(strategy="mean", add_indicator=True, column_missing_threshold=0.5)
    out = step.fit_transform(df)
    assert "b" not in out.columns  # dropped
    assert "b__missing" in out.columns  # but its missingness is preserved
    assert "b__missing" in list(step.get_feature_names_out())
