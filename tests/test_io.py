"""Tests for nextaire_tools.io tabular loading/saving."""

from __future__ import annotations

import pandas as pd
import pytest

from nextaire_tools.exceptions import SchemaError
from nextaire_tools.io import load_table, save_table


def test_load_csv_roundtrip(aq_csv):
    df = load_table(aq_csv)
    assert isinstance(df, pd.DataFrame)
    assert {"no2", "o3", "pm10"}.issubset(df.columns)
    assert len(df) == 720


def test_load_csv_with_time_index(aq_csv):
    df = load_table(aq_csv, time_col="timestamp", set_time_index=True)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert "timestamp" not in df.columns


@pytest.mark.parametrize("suffix", [".parquet", ".xlsx"])
def test_save_and_load_other_formats(tmp_path, aq_df, suffix):
    path = tmp_path / f"data{suffix}"
    save_table(aq_df, path)
    assert path.exists()
    reloaded = load_table(path)
    assert reloaded.shape[1] >= 3


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_table(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_table("does-not-exist.csv")


def test_load_column_subset(aq_csv):
    df = load_table(aq_csv, columns=["timestamp", "no2"])
    assert list(df.columns) == ["timestamp", "no2"]
