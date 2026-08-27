"""Time-series-aware cross-validation splitters for :mod:`nextaire_tools`.

Ordinary :class:`~sklearn.model_selection.KFold` shuffles rows, which leaks
future information into the training fold and inflates scores on temporally
correlated air-quality series. The splitters here all respect chronological
order: a test block never precedes its training block, and an optional ``gap``
can be inserted between the two to emulate a forecasting horizon.

The module re-exports scikit-learn's
:class:`~sklearn.model_selection.TimeSeriesSplit` (expanding-window folds that
share a common origin) so callers can obtain every splitter from a single
namespace, and adds three complementary strategies:

* :class:`BlockingTimeSeriesSplit` -- independent, non-overlapping blocks.
* :class:`SlidingWindowSplit` -- a fixed-width training window sliding forward.
* :class:`ExpandingWindowSplit` -- a growing training window.

Every splitter exposes the scikit-learn ``split`` / ``get_n_splits`` protocol,
so they drop into :func:`nextaire_tools.models.cross_val_report` and scikit-learn's own
cross-validation helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.utils.logging import get_logger

__all__ = [
    "BlockingTimeSeriesSplit",
    "ExpandingWindowSplit",
    "SlidingWindowSplit",
    "TimeSeriesSplit",
    "temporal_train_test_split",
]

_LOG = get_logger(__name__)

# Anything the splitters accept as the sample container ``X``.
ArrayLike: TypeAlias = np.ndarray | pd.DataFrame | pd.Series | list | tuple


def _num_samples(x: ArrayLike) -> int:
    """Return the number of samples (rows) in ``x``.

    Uses ``shape[0]`` when available (arrays / DataFrames) and falls back to
    :func:`len` otherwise.
    """
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(x)


class BlockingTimeSeriesSplit:
    """Cross-validator yielding independent, non-overlapping time blocks.

    The series is divided into ``n_splits`` contiguous blocks of equal size
    (the final block absorbs any remainder). Within each block the earlier half
    forms the training set and the later half the test set, so folds never
    share samples -- unlike :class:`~sklearn.model_selection.TimeSeriesSplit`,
    whose training set grows and overlaps across folds. This is useful when the
    series is long enough that distant history is uninformative and each block
    should be evaluated on its own footing.

    Parameters
    ----------
    n_splits:
        Number of blocks / folds. Must be ``>= 2``.
    gap:
        Number of samples dropped between the train and test halves of each
        block (emulates a forecast lead time). Must be ``>= 0``.

    Raises
    ------
    ConfigurationError
        If ``n_splits < 2`` or ``gap < 0``.

    Examples
    --------
    >>> import numpy as np
    >>> cv = BlockingTimeSeriesSplit(n_splits=3)
    >>> for train, test in cv.split(np.arange(30)):
    ...     assert train.max() < test.min()  # no leakage
    """

    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        if n_splits < 2:
            raise ConfigurationError(f"n_splits must be >= 2, got {n_splits}.")
        if gap < 0:
            raise ConfigurationError(f"gap must be >= 0, got {gap}.")
        self.n_splits = n_splits
        self.gap = gap

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_index, test_index)`` pairs, one per block.

        Parameters
        ----------
        X:
            Training data used only to infer the number of samples.
        y, groups:
            Ignored; present for scikit-learn API compatibility.

        Yields
        ------
        tuple of numpy.ndarray
            Integer position arrays for the train and test sets of each block.

        Raises
        ------
        ConfigurationError
            If there are too few samples for the requested configuration.
        """
        n = _num_samples(X)
        if self.n_splits > n:
            raise ConfigurationError(f"n_splits={self.n_splits} cannot exceed n_samples={n}.")
        fold_size = n // self.n_splits
        if fold_size < 2:
            raise ConfigurationError(
                f"Each block needs >= 2 samples (got block size {fold_size}); reduce n_splits."
            )
        indices = np.arange(n)
        for i in range(self.n_splits):
            start = i * fold_size
            stop = n if i == self.n_splits - 1 else start + fold_size
            block = stop - start
            n_train = (block - self.gap) // 2
            if n_train < 1 or (block - n_train - self.gap) < 1:
                raise ConfigurationError(
                    "Block too small for the requested gap; reduce gap or n_splits."
                )
            train_end = start + n_train
            test_start = train_end + self.gap
            yield indices[start:train_end], indices[test_start:stop]

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splitting iterations (``n_splits``)."""
        return self.n_splits


class SlidingWindowSplit:
    """Rolling-origin cross-validator with a fixed-width training window.

    Both the training and test windows have a constant length; the origin
    advances by ``step`` samples each fold. Because old samples fall out of the
    window as new ones enter, this evaluates how well a model tracks a possibly
    non-stationary series using only recent history.

    Parameters
    ----------
    train_size:
        Number of samples in each (fixed-width) training window. ``>= 1``.
    test_size:
        Number of samples in each test window. ``>= 1``.
    step:
        Advance of the window between folds. Defaults to ``test_size`` (giving
        contiguous, non-overlapping test windows). ``>= 1`` when given.
    gap:
        Samples dropped between the train and test windows. ``>= 0``.

    Raises
    ------
    ConfigurationError
        If any size is non-positive or ``gap < 0``.

    Examples
    --------
    >>> import numpy as np
    >>> cv = SlidingWindowSplit(train_size=10, test_size=5)
    >>> folds = list(cv.split(np.arange(30)))
    >>> all(len(tr) == 10 for tr, _ in folds)
    True
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        *,
        step: int | None = None,
        gap: int = 0,
    ) -> None:
        if train_size < 1:
            raise ConfigurationError(f"train_size must be >= 1, got {train_size}.")
        if test_size < 1:
            raise ConfigurationError(f"test_size must be >= 1, got {test_size}.")
        if step is not None and step < 1:
            raise ConfigurationError(f"step must be >= 1, got {step}.")
        if gap < 0:
            raise ConfigurationError(f"gap must be >= 0, got {gap}.")
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.gap = gap

    def _make_folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(n)
        step = self.step if self.step is not None else self.test_size
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        train_start = 0
        while True:
            train_end = train_start + self.train_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            if test_end > n:
                break
            folds.append((indices[train_start:train_end], indices[test_start:test_end]))
            train_start += step
        return folds

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_index, test_index)`` pairs for each window.

        Raises
        ------
        ConfigurationError
            If no fold fits within the available samples.
        """
        folds = self._make_folds(_num_samples(X))
        if not folds:
            raise ConfigurationError(
                "Not enough samples for the requested sliding-window configuration."
            )
        yield from folds

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of folds; requires ``X`` to know ``n_samples``."""
        if X is None:
            raise ConfigurationError("get_n_splits requires X to determine the number of folds.")
        return len(self._make_folds(_num_samples(X)))


class ExpandingWindowSplit:
    """Rolling-origin cross-validator with a growing training window.

    The training window always starts at the first sample and grows by ``step``
    each fold, while a fixed-width test window rolls forward just after it. This
    mirrors the common production setting where a model is periodically
    retrained on all history accumulated so far.

    Parameters
    ----------
    initial_train_size:
        Number of samples in the first training window. ``>= 1``.
    test_size:
        Number of samples in each test window. ``>= 1``.
    step:
        Growth of the training window between folds. Defaults to ``test_size``.
        ``>= 1`` when given.
    gap:
        Samples dropped between the train and test windows. ``>= 0``.

    Raises
    ------
    ConfigurationError
        If any size is non-positive or ``gap < 0``.

    Examples
    --------
    >>> import numpy as np
    >>> cv = ExpandingWindowSplit(initial_train_size=10, test_size=5)
    >>> tr0, _ = next(cv.split(np.arange(30)))
    >>> len(tr0)
    10
    """

    def __init__(
        self,
        initial_train_size: int,
        test_size: int,
        *,
        step: int | None = None,
        gap: int = 0,
    ) -> None:
        if initial_train_size < 1:
            raise ConfigurationError(f"initial_train_size must be >= 1, got {initial_train_size}.")
        if test_size < 1:
            raise ConfigurationError(f"test_size must be >= 1, got {test_size}.")
        if step is not None and step < 1:
            raise ConfigurationError(f"step must be >= 1, got {step}.")
        if gap < 0:
            raise ConfigurationError(f"gap must be >= 0, got {gap}.")
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step = step
        self.gap = gap

    def _make_folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(n)
        step = self.step if self.step is not None else self.test_size
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        train_end = self.initial_train_size
        while True:
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            if test_end > n:
                break
            folds.append((indices[0:train_end], indices[test_start:test_end]))
            train_end += step
        return folds

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_index, test_index)`` pairs for each fold.

        Raises
        ------
        ConfigurationError
            If no fold fits within the available samples.
        """
        folds = self._make_folds(_num_samples(X))
        if not folds:
            raise ConfigurationError(
                "Not enough samples for the requested expanding-window configuration."
            )
        yield from folds

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of folds; requires ``X`` to know ``n_samples``."""
        if X is None:
            raise ConfigurationError("get_n_splits requires X to determine the number of folds.")
        return len(self._make_folds(_num_samples(X)))


def temporal_train_test_split(
    data: ArrayLike,
    *,
    test_size: float | int = 0.2,
    gap: int = 0,
) -> tuple[ArrayLike, ArrayLike]:
    """Split a time-ordered dataset into train and test parts without shuffling.

    The last ``test_size`` samples become the test set and the earlier samples
    the training set, preserving chronological order. Unlike
    :func:`sklearn.model_selection.train_test_split`, no shuffling occurs, so no
    future information leaks into training. An optional ``gap`` drops the
    samples immediately preceding the test set (a forecast lead time).

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame`, :class:`pandas.Series`, NumPy array, list,
        or tuple. The two returned objects have the same type as ``data``.
    test_size:
        Test-set size. A ``float`` in ``(0, 1)`` is a fraction of the samples
        (rounded to the nearest integer); an ``int`` is an explicit count.
    gap:
        Number of samples discarded between the train and test sets. ``>= 0``.

    Returns
    -------
    tuple
        ``(train, test)``, each the same type as ``data``.

    Raises
    ------
    ConfigurationError
        If ``test_size`` is out of range, ``gap`` is negative, or the split
        leaves no training samples.

    Examples
    --------
    >>> import numpy as np
    >>> train, test = temporal_train_test_split(np.arange(10), test_size=0.3)
    >>> train, test
    (array([0, 1, 2, 3, 4, 5, 6]), array([7, 8, 9]))
    """
    if gap < 0:
        raise ConfigurationError(f"gap must be >= 0, got {gap}.")
    n = _num_samples(data)
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ConfigurationError(f"Float test_size must be in (0, 1), got {test_size}.")
        n_test = round(test_size * n)
    else:
        n_test = int(test_size)
    if not 0 < n_test < n:
        raise ConfigurationError(f"Resolved test size {n_test} is out of range for {n} samples.")
    test_start = n - n_test
    train_end = test_start - gap
    if train_end < 1:
        raise ConfigurationError("Split leaves no training samples; reduce test_size or gap.")

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[:train_end], data.iloc[test_start:]
    return data[:train_end], data[test_start:]
