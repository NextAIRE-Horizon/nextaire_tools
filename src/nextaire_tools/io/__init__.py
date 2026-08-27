"""Tabular IO: read/write CSV, Excel, and Parquet with datetime handling."""

from __future__ import annotations

from nextaire_tools.io.readers import SUPPORTED_SUFFIXES, load_table, save_table

__all__ = ["load_table", "save_table", "SUPPORTED_SUFFIXES"]
