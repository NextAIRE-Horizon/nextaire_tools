"""Shared type aliases for :mod:`nextaire_tools`.

Kept in a private module so both the public API and internal helpers can import
the same names without circular-import risk.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from os import PathLike
from typing import Literal

# A single column label or a list of them.
ColumnLike = Hashable | Sequence[Hashable]

# Anything acceptable as a filesystem path.
PathType = str | PathLike[str]

# How a step should select the columns it operates on when none are given.
ColumnSelector = Literal["numeric", "all"]

__all__ = ["ColumnLike", "PathType", "ColumnSelector"]
