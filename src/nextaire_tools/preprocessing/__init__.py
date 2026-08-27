"""Preprocessing steps: DataFrame-in/DataFrame-out, scikit-learn-compatible.

Each step subclasses :class:`~nextaire_tools.preprocessing.base.BaseStep` and can be used
standalone, inside :class:`~nextaire_tools.preprocessing.pipeline.Pipeline`, or inside a
:class:`sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.preprocessing.features import CorrelationFilter, LagFeatures, WindDecomposer
from nextaire_tools.preprocessing.missing import MissingValueHandler
from nextaire_tools.preprocessing.outliers import OutlierHandler
from nextaire_tools.preprocessing.pipeline import STEP_REGISTRY, Pipeline, make_pipeline
from nextaire_tools.preprocessing.scaling import Scaler
from nextaire_tools.preprocessing.temporal import TemporalFeatures

__all__ = [
    "BaseStep",
    "MissingValueHandler",
    "OutlierHandler",
    "TemporalFeatures",
    "WindDecomposer",
    "LagFeatures",
    "CorrelationFilter",
    "Scaler",
    "Pipeline",
    "make_pipeline",
    "STEP_REGISTRY",
]
