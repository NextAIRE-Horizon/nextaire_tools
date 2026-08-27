"""The base class for every :mod:`nextaire_tools` preprocessing step.

Design contract
---------------
A *step* is a scikit-learn-compatible transformer that **consumes and produces a
:class:`pandas.DataFrame`**. This keeps column names, dtypes, and the
(datetime) index intact through an entire pipeline — unlike bare NumPy
transformers, which discard that metadata.

Subclasses implement two protected hooks:

* :meth:`_fit(X, y)` — learn any state (store it in ``trailing_underscore_``
  attributes) and return ``None``.
* :meth:`_transform(X)` — return a **new** DataFrame.

Everything else (validation, the fitted-state guard, ``get_params`` /
``set_params``, ``set_output`` compatibility) is handled here.

.. note::
   Some steps (outlier and missing-value removal in ``"drop"`` mode) change the
   **number of rows**. That is intentional and works within :class:`nextaire_tools.Pipeline`,
   which flows a single DataFrame. It is, however, incompatible with
   :class:`sklearn.pipeline.Pipeline` when a separate ``y`` is supplied, because
   ``X`` and ``y`` would fall out of alignment. Keep the target as a column of the
   DataFrame (the nextaire_tools convention) to stay aligned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nextaire_tools._typing import ColumnLike
from nextaire_tools.exceptions import NotFittedError
from nextaire_tools.utils.validation import check_dataframe, resolve_columns

__all__ = ["BaseStep"]


class BaseStep(TransformerMixin, BaseEstimator, ABC):
    """Abstract, DataFrame-in / DataFrame-out transformer.

    Attributes set after :meth:`fit`
    ---------------------------------
    feature_names_in_ : numpy.ndarray
        Column labels seen during ``fit``.
    n_features_in_ : int
        Number of columns seen during ``fit``.
    """

    # Subclasses that operate on a user-selectable subset of columns should set
    # this to the *attribute name* holding the selection (e.g. ``"columns"``) so
    # the base class can resolve it uniformly. ``None`` means "the whole frame".
    _column_param: str | None = "columns"
    # When resolving default columns, restrict to numeric dtypes?
    _numeric_only: bool = False

    # ------------------------------------------------------------------ API
    def fit(self, X: pd.DataFrame, y: object = None) -> BaseStep:
        """Learn state from ``X``.

        Parameters
        ----------
        X:
            Input frame.
        y:
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        self
        """
        Xdf = check_dataframe(X, copy=True)
        self._validate_params()
        self.feature_names_in_ = np.asarray(Xdf.columns, dtype=object)
        self.n_features_in_ = Xdf.shape[1]
        self.columns_ = self._resolve_columns(Xdf)
        self._fit(Xdf, y)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to ``X``.

        Returns
        -------
        pandas.DataFrame
            A new frame; the input is never mutated.
        """
        self._check_is_fitted()
        Xdf = check_dataframe(X, copy=True)
        out = self._transform(Xdf)
        if not isinstance(out, pd.DataFrame):  # defensive: subclass contract
            raise TypeError(
                f"{type(self).__name__}._transform must return a DataFrame, "
                f"got {type(out).__name__}."
            )
        return out

    def fit_transform(
        self, X: pd.DataFrame, y: object = None, **fit_params: object
    ) -> pd.DataFrame:
        """Fit then transform in one call (avoids re-validating twice)."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names.

        The base implementation assumes columns are unchanged. Steps that add or
        remove columns override this.
        """
        self._check_is_fitted()
        return np.asarray(self.feature_names_in_, dtype=object)

    # ---------------------------------------------------------- subclass hooks
    def _fit(self, X: pd.DataFrame, y: object = None) -> None:
        """Learn state. Default: nothing to learn."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed frame. Must be implemented by subclasses."""

    def _validate_params(self) -> None:
        """Validate constructor parameters. Override to add checks."""

    # ------------------------------------------------------------- internals
    def _resolve_columns(self, X: pd.DataFrame) -> list[Hashable]:
        """Resolve the step's target columns from its ``_column_param``."""
        if self._column_param is None:
            return list(X.columns)
        selection: ColumnLike | None = getattr(self, self._column_param, None)
        return resolve_columns(X, selection, numeric_only=self._numeric_only)

    def _check_is_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' (or 'fit_transform') before using this step."
            )
