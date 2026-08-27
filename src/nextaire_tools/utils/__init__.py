"""Shared utilities: validation, optional-dependency handling, and logging."""

from __future__ import annotations

from nextaire_tools.utils.logging import enable_logging, get_logger
from nextaire_tools.utils.validation import (
    check_dataframe,
    ensure_datetime_index,
    require,
    resolve_columns,
)

__all__ = [
    "enable_logging",
    "get_logger",
    "check_dataframe",
    "ensure_datetime_index",
    "require",
    "resolve_columns",
]
