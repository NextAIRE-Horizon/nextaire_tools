"""A lightweight, DataFrame-native preprocessing pipeline.

This module provides :class:`Pipeline`, a minimal composition primitive that
threads a single :class:`pandas.DataFrame` through an ordered list of
:class:`~nextaire_tools.preprocessing.base.BaseStep` transformers. Unlike
:class:`sklearn.pipeline.Pipeline`, it flows one frame (never a separate ``X`` /
``y`` pair), which keeps steps that change the number of rows — such as dropping
outliers or rows with missing values — perfectly aligned.

Pipelines can be built three ways:

* directly, from a list of steps or ``(name, step)`` tuples;
* with the :func:`make_pipeline` helper, which auto-names steps; or
* declaratively, from a list of config dicts via :meth:`Pipeline.from_config`,
  which resolves step names through the module-level :data:`STEP_REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from nextaire_tools.exceptions import ConfigurationError, NotFittedError
from nextaire_tools.preprocessing.base import BaseStep
from nextaire_tools.preprocessing.features import CorrelationFilter, LagFeatures, WindDecomposer
from nextaire_tools.preprocessing.missing import MissingValueHandler
from nextaire_tools.preprocessing.outliers import OutlierHandler
from nextaire_tools.preprocessing.scaling import Scaler
from nextaire_tools.preprocessing.temporal import TemporalFeatures
from nextaire_tools.utils.logging import get_logger

__all__ = ["STEP_REGISTRY", "Pipeline", "make_pipeline"]

_LOG = get_logger(__name__)

# Maps the string used in :meth:`Pipeline.from_config` to the step class.
STEP_REGISTRY: dict[str, type] = {
    "MissingValueHandler": MissingValueHandler,
    "OutlierHandler": OutlierHandler,
    "TemporalFeatures": TemporalFeatures,
    "WindDecomposer": WindDecomposer,
    "LagFeatures": LagFeatures,
    "CorrelationFilter": CorrelationFilter,
    "Scaler": Scaler,
}


class Pipeline:
    """Chain preprocessing steps into a single DataFrame-in / DataFrame-out unit.

    Parameters
    ----------
    steps : sequence
        Ordered steps. Each element is either a
        :class:`~nextaire_tools.preprocessing.base.BaseStep` instance (auto-named) or a
        ``(name, step)`` tuple.

    Attributes
    ----------
    steps_ : list of tuple
        The normalized ``(name, step)`` pairs.

    Raises
    ------
    ConfigurationError
        If ``steps`` is empty, contains a non-:class:`BaseStep` object, or has
        duplicate step names.

    Examples
    --------
    >>> from nextaire_tools.preprocessing import MissingValueHandler, Scaler
    >>> pipe = Pipeline([MissingValueHandler(strategy="mean"), Scaler()])
    >>> len(pipe)
    2
    """

    def __init__(self, steps: Sequence[Any]) -> None:
        self.steps = steps
        self.steps_ = self._normalize_steps(steps)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalize_steps(steps: Sequence[Any]) -> list[tuple[str, BaseStep]]:
        if not isinstance(steps, Iterable):
            raise ConfigurationError("steps must be an iterable of steps.")
        materialized = list(steps)
        if not materialized:
            raise ConfigurationError("Pipeline requires at least one step.")

        normalized: list[tuple[str, BaseStep]] = []
        seen: set[str] = set()
        for i, item in enumerate(materialized):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, step = item
            else:
                step = item
                name = f"{type(step).__name__.lower()}{i}"
            if not isinstance(step, BaseStep):
                raise ConfigurationError(
                    f"Pipeline steps must be BaseStep instances, got "
                    f"{type(step).__name__!r} at position {i}."
                )
            if name in seen:
                raise ConfigurationError(f"Duplicate step name {name!r}.")
            seen.add(name)
            normalized.append((name, step))
        return normalized

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise NotFittedError(
                "This Pipeline is not fitted yet. Call 'fit' or 'fit_transform' first."
            )

    # ------------------------------------------------------------------- API
    def fit(self, X: pd.DataFrame, y: object = None) -> Pipeline:
        """Fit every step in sequence, threading the transformed frame.

        Parameters
        ----------
        X : pandas.DataFrame
            Input frame.
        y : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        Pipeline
            The fitted pipeline (``self``).
        """
        Xt = X
        for _name, step in self.steps_:
            Xt = step.fit_transform(Xt, y)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted step in sequence.

        Parameters
        ----------
        X : pandas.DataFrame
            Input frame.

        Returns
        -------
        pandas.DataFrame
            The transformed frame; the input is never mutated.
        """
        self._check_fitted()
        Xt = X
        for _name, step in self.steps_:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit and transform in a single pass.

        Parameters
        ----------
        X : pandas.DataFrame
            Input frame.
        y : object, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        pandas.DataFrame
            The transformed frame.
        """
        Xt = X
        for _name, step in self.steps_:
            Xt = step.fit_transform(Xt, y)
        self.fitted_ = True
        return Xt

    def get_feature_names_out(self, input_features: object = None) -> Any:
        """Delegate to the last step's ``get_feature_names_out``.

        Parameters
        ----------
        input_features : object, optional
            Passed through to the final step.

        Returns
        -------
        numpy.ndarray
            The output feature names reported by the final step.

        Raises
        ------
        AttributeError
            If the final step does not implement ``get_feature_names_out``.
        """
        self._check_fitted()
        last = self.steps_[-1][1]
        if not hasattr(last, "get_feature_names_out"):
            raise AttributeError(
                f"The final step {type(last).__name__!r} has no get_feature_names_out."
            )
        return last.get_feature_names_out(input_features)

    # ------------------------------------------------------------ properties
    @property
    def named_steps(self) -> dict[str, BaseStep]:
        """Mapping of step name to step instance."""
        return dict(self.steps_)

    # -------------------------------------------------------- dunder helpers
    def __getitem__(self, key: int | str | slice) -> BaseStep | Pipeline:
        if isinstance(key, slice):
            return Pipeline(self.steps_[key])
        if isinstance(key, str):
            return self.named_steps[key]
        if isinstance(key, int):
            return self.steps_[key][1]
        raise TypeError(f"Pipeline indices must be int, str, or slice, not {type(key).__name__}.")

    def __len__(self) -> int:
        return len(self.steps_)

    def __repr__(self) -> str:
        body = ",\n  ".join(f"({name!r}, {step!r})" for name, step in self.steps_)
        return f"Pipeline(steps=[\n  {body}\n])"

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: Sequence[Mapping[str, Any]]) -> Pipeline:
        """Build a pipeline from a declarative list of config dicts.

        Parameters
        ----------
        config : sequence of mapping
            Each entry must have a ``"step"`` key naming a class in
            :data:`STEP_REGISTRY` and may have a ``"params"`` mapping and an
            optional ``"name"``.

        Returns
        -------
        Pipeline

        Raises
        ------
        ConfigurationError
            If an entry is malformed or names an unknown step.

        Examples
        --------
        >>> config = [{"step": "MissingValueHandler", "params": {"strategy": "mean"}}]
        >>> pipe = Pipeline.from_config(config)
        >>> len(pipe)
        1
        """
        if not isinstance(config, Iterable):
            raise ConfigurationError("config must be an iterable of step mappings.")

        steps: list[Any] = []
        for i, entry in enumerate(config):
            if not isinstance(entry, Mapping) or "step" not in entry:
                raise ConfigurationError(
                    f"Config entry {i} must be a mapping with a 'step' key, got {entry!r}."
                )
            step_name = entry["step"]
            step_cls = STEP_REGISTRY.get(step_name)
            if step_cls is None:
                raise ConfigurationError(
                    f"Unknown step {step_name!r}. Available: {sorted(STEP_REGISTRY)}."
                )
            params = entry.get("params") or {}
            step = step_cls(**params)
            name = entry.get("name")
            steps.append((name, step) if name else step)
        return cls(steps)


def make_pipeline(*steps: BaseStep) -> Pipeline:
    """Build a :class:`Pipeline`, auto-naming each step.

    Parameters
    ----------
    *steps : BaseStep
        The steps to chain, in order. Each is named by its lowercased class name
        suffixed with its position index (e.g. ``"scaler1"``).

    Returns
    -------
    Pipeline

    Examples
    --------
    >>> from nextaire_tools.preprocessing import MissingValueHandler, Scaler
    >>> pipe = make_pipeline(MissingValueHandler(strategy="mean"), Scaler())
    >>> sorted(pipe.named_steps)
    ['missingvaluehandler0', 'scaler1']
    """
    return Pipeline(list(steps))
