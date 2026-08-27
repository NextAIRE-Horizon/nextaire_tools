"""Non-negative matrix factorisation for source apportionment.

Source apportionment attributes measured pollutant concentrations to a small
number of latent emission sources ("factors"). Non-negative matrix factorisation
(NMF) decomposes the non-negative sample x pollutant matrix ``V`` into
``V ~= W @ H`` with ``W >= 0`` (sample x factor contributions) and ``H >= 0``
(factor x pollutant profiles), which mirrors the additive, non-negative physics
of receptor models such as PMF.

scikit-learn is a hard dependency of the package, so
:class:`~sklearn.decomposition.NMF` is imported normally.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from nextaire_tools.exceptions import ConfigurationError
from nextaire_tools.utils.logging import get_logger

__all__ = ["NMFApportionment"]

_LOG = get_logger(__name__)


class NMFApportionment(BaseEstimator):
    """NMF factor model for pollutant source apportionment.

    Parameters
    ----------
    n_components:
        Number of latent source factors to extract.
    scale:
        MinMax-scale each column to ``[0, 1]`` before factorisation. NMF requires
        non-negative input; scaling also puts pollutants on a common range so no
        single high-concentration species dominates the fit.
    random_state:
        Seed forwarded to :class:`~sklearn.decomposition.NMF` for reproducibility.
    **nmf_kwargs:
        Extra keyword arguments forwarded to :class:`~sklearn.decomposition.NMF`
        (e.g. ``max_iter``, ``l1_ratio``, ``alpha_W``).

    Attributes
    ----------
    components_ : pandas.DataFrame
        The ``H`` matrix (factor x pollutant), indexed ``factor_1 .. factor_k``
        with the original pollutant columns as labels.
    feature_names_in_ : numpy.ndarray
        Pollutant column labels seen at ``fit``.
    nmf_ : sklearn.decomposition.NMF
        The fitted scikit-learn estimator.
    scaler_ : sklearn.preprocessing.MinMaxScaler or None
        The fitted scaler (``None`` when ``scale`` is ``False``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.random((50, 4)), columns=list("abcd"))
    >>> model = NMFApportionment(n_components=2).fit(X)
    >>> W = model.transform(X)
    >>> list(W.columns)
    ['factor_1', 'factor_2']
    >>> model.components_.shape
    (2, 4)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        random_state: int = 0,
        **nmf_kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.nmf_kwargs = nmf_kwargs

    # --------------------------------------------------------------- API
    def fit(self, X: pd.DataFrame, y: object = None) -> NMFApportionment:
        """Fit the NMF model on ``X``.

        Parameters
        ----------
        X:
            DataFrame of non-negative concentrations (pollutants as columns,
            samples as rows).
        y:
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        self

        Raises
        ------
        ConfigurationError
            If ``X`` is not a DataFrame, contains NaNs, or (when ``scale`` is
            ``False``) contains negative values.
        """
        self._fit_matrix(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the ``W`` matrix (samples x factors) as a DataFrame.

        Parameters
        ----------
        X:
            DataFrame with the same pollutant columns seen at ``fit``.

        Returns
        -------
        pandas.DataFrame
            Indexed like ``X`` with columns ``factor_1 .. factor_k``.
        """
        if not hasattr(self, "nmf_"):
            raise ConfigurationError("NMFApportionment is not fitted yet. Call 'fit' first.")
        values = self._prepare(X, fitting=False)
        W = self.nmf_.transform(values)
        return pd.DataFrame(W, index=X.index, columns=self._factor_names())

    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit the model and return the ``W`` matrix in one call."""
        values = self._fit_matrix(X)
        W = self.nmf_.transform(values)
        return pd.DataFrame(W, index=X.index, columns=self._factor_names())

    def loadings(self) -> pd.DataFrame:
        """Return the factor x pollutant loadings (the ``H`` matrix).

        Returns
        -------
        pandas.DataFrame
            The :attr:`components_` frame, for interpretation / biplots.
        """
        if not hasattr(self, "components_"):
            raise ConfigurationError("NMFApportionment is not fitted yet. Call 'fit' first.")
        return self.components_

    # --------------------------------------------------------- internals
    def _fit_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Validate, scale, and fit NMF; return the transformed input matrix."""
        values = self._prepare(X, fitting=True)
        nmf = NMF(
            n_components=int(self.n_components),
            random_state=self.random_state,
            **self.nmf_kwargs,
        )
        nmf.fit(values)
        self.nmf_ = nmf
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.components_ = pd.DataFrame(
            nmf.components_,
            index=self._factor_names(),
            columns=list(X.columns),
        )
        _LOG.info(
            "Fitted NMFApportionment: %d factors over %d pollutants.",
            int(self.n_components),
            X.shape[1],
        )
        return values

    def _prepare(self, X: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        """Validate ``X`` and return the (optionally scaled) numeric matrix."""
        if not isinstance(X, pd.DataFrame):
            raise ConfigurationError(f"X must be a pandas.DataFrame, got {type(X).__name__!r}.")
        arr = X.to_numpy(dtype=float)
        if np.isnan(arr).any():
            raise ConfigurationError(
                "X contains NaN values; NMF requires a complete non-negative matrix. "
                "Impute or drop missing values first."
            )
        if self.scale:
            if fitting:
                self.scaler_ = MinMaxScaler()
                return self.scaler_.fit_transform(arr)
            return self.scaler_.transform(arr)
        self.scaler_ = None
        if (arr < 0).any():
            raise ConfigurationError(
                "X contains negative values but scale=False; NMF requires "
                "non-negative input. Set scale=True or provide non-negative data."
            )
        return arr

    def _factor_names(self) -> list[str]:
        return [f"factor_{i + 1}" for i in range(int(self.n_components))]
