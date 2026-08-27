"""Tests for time-series cross-validation splitters (leakage-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextaire_tools.models import (
    BlockingTimeSeriesSplit,
    ExpandingWindowSplit,
    SlidingWindowSplit,
    temporal_train_test_split,
)

N = 200
X = np.arange(N).reshape(-1, 1)


def _assert_no_leakage(splitter, gap=0):
    n = 0
    for train_idx, test_idx in splitter.split(X):
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert train_idx.max() < test_idx.min()  # train strictly precedes test
        assert train_idx.max() + gap < test_idx.min() + 1
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        n += 1
    assert n == splitter.get_n_splits(X)


def test_blocking_split_no_leakage():
    _assert_no_leakage(BlockingTimeSeriesSplit(n_splits=5))


def test_sliding_window_fixed_train_size():
    sp = SlidingWindowSplit(train_size=50, test_size=25)
    sizes = {len(tr) for tr, _ in sp.split(X)}
    assert sizes == {50}
    _assert_no_leakage(sp)


def test_expanding_window_growing_train():
    sp = ExpandingWindowSplit(initial_train_size=50, test_size=25)
    train_lengths = [len(tr) for tr, _ in sp.split(X)]
    assert train_lengths == sorted(train_lengths)
    assert train_lengths[0] == 50
    _assert_no_leakage(sp)


def test_gap_is_respected():
    sp = SlidingWindowSplit(train_size=40, test_size=20, gap=5)
    for train_idx, test_idx in sp.split(X):
        assert test_idx.min() - train_idx.max() > 5


def test_temporal_train_test_split_preserves_order():
    df = pd.DataFrame(
        {"v": np.arange(100)}, index=pd.date_range("2024-01-01", periods=100, freq="h")
    )
    train, test = temporal_train_test_split(df, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train.index.max() < test.index.min()


def test_splitter_validates_sample_count():
    with pytest.raises((ValueError, Exception)):
        list(SlidingWindowSplit(train_size=500, test_size=500).split(X))
