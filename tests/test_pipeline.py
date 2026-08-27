"""Tests for the preprocessing Pipeline."""

from __future__ import annotations

import pandas as pd

from nextaire_tools.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    Pipeline,
    Scaler,
    TemporalFeatures,
    make_pipeline,
)


def test_pipeline_chains_steps(aq_df):
    pipe = Pipeline(
        [
            MissingValueHandler(strategy="interpolate", limit=3),
            OutlierHandler(columns=["no2", "o3", "pm10"], method="iqr", strategy="clip"),
            TemporalFeatures(add=("hour",), cyclical=("hour", "dayofyear")),
            Scaler(columns=["no2", "o3", "pm10"], method="standard"),
        ]
    )
    out = pipe.fit_transform(aq_df)
    assert isinstance(out, pd.DataFrame)
    assert "hour_sin" in out.columns
    assert out[["no2", "o3", "pm10"]].isna().to_numpy().sum() == 0


def test_named_steps_and_indexing(aq_df):
    pipe = make_pipeline(MissingValueHandler(strategy="mean"), Scaler())
    pipe.fit(aq_df.assign())
    assert len(pipe) == 2
    assert isinstance(pipe[0], MissingValueHandler)
    names = list(pipe.named_steps)
    assert pipe[names[0]] is pipe[0]


def test_from_config(aq_df):
    config = [
        {"step": "MissingValueHandler", "params": {"strategy": "mean"}},
        {"step": "TemporalFeatures", "params": {"add": ["hour"], "cyclical": ["hour"]}},
    ]
    pipe = Pipeline.from_config(config)
    out = pipe.fit_transform(aq_df)
    assert "hour_sin" in out.columns


def test_fit_then_transform_equivalent(aq_df):
    pipe = make_pipeline(MissingValueHandler(strategy="mean"))
    a = pipe.fit_transform(aq_df)
    b = pipe.fit(aq_df).transform(aq_df)
    pd.testing.assert_frame_equal(a, b)
