"""PyTorch deep-learning regressors for hourly air-pollutant forecasting.

This module implements the neural architectures used in the author's hourly
air-quality forecasting work -- a multilayer perceptron (:class:`MLPRegressor`),
a recurrent network (:class:`LSTMRegressor`), and a 1-D convolutional network
(:class:`CNNRegressor`) -- behind a small, scikit-learn-style ``fit`` /
``predict`` API so they interchange with the classical models in
:mod:`nextaire_tools.models.sklearn_models` and flow through
:func:`nextaire_tools.models.cross_val_report`.

PyTorch is an **optional** dependency (the ``deep`` extra). It is imported
lazily inside the methods that need it via
:func:`nextaire_tools.utils.validation.require`, so importing this module with only the
core dependencies installed always succeeds.

Design notes
------------
* Inputs and targets are standardised internally (mean/std stored at ``fit``),
  which makes training robust and removes the need for the caller to scale.
* The recurrent and convolutional models consume sliding windows built with
  :func:`make_sequences`. Their :meth:`predict` returns one value per input row
  and fills the first ``window`` rows -- which lack a full look-back window --
  with ``NaN`` so the output length always equals ``len(X)``.
* Training uses Adam + MSE with early stopping on a chronological tail
  validation split, and is deterministic given ``random_state`` on CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from nextaire_tools.exceptions import ConfigurationError, NotFittedError
from nextaire_tools.utils.logging import get_logger
from nextaire_tools.utils.validation import require

__all__ = [
    "CNNRegressor",
    "LSTMRegressor",
    "MLPRegressor",
    "make_sequences",
]

_LOG = get_logger(__name__)


def _torch() -> Any:
    """Import and return the optional :mod:`torch` module (``deep`` extra)."""
    return require("torch", "deep", "deep-learning models (MLP/LSTM/CNN)")


def make_sequences(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    window: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build sliding time windows from a 2-D feature matrix.

    Each window covers ``window`` consecutive rows, ``X[k : k + window]``. When
    ``y`` is given, the target for window ``k`` is the value ``horizon`` steps
    after the window's last row, i.e. ``y[k + window - 1 + horizon]``. The
    number of windows is ``n_samples - window - horizon + 1`` regardless of
    whether ``y`` is supplied, so the windows produced for prediction align
    exactly with those produced for training.

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)``.
    y:
        Optional 1-D target array of length ``n_samples``. When ``None`` only
        the windows are returned.
    window:
        Look-back length, ``>= 1``.
    horizon:
        Forecast lead time in steps, ``>= 1``. ``1`` predicts the next value.

    Returns
    -------
    tuple
        ``(X3d, y2d)`` where ``X3d`` has shape
        ``(n_windows, window, n_features)`` and ``y2d`` has shape
        ``(n_windows, 1)`` (or ``None`` when ``y`` is not given).

    Raises
    ------
    ConfigurationError
        If ``X`` is not 2-D, ``window`` or ``horizon`` is below 1, ``y`` has the
        wrong length, or there are too few samples to form a single window.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> X3d, y2d = make_sequences(X, X.ravel(), window=3, horizon=1)
    >>> X3d.shape, y2d.shape
    ((7, 3, 1), (7, 1))
    >>> y2d[0, 0]  # value one step after the first window (rows 0,1,2)
    3.0
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ConfigurationError(
            f"X must be 2-D (n_samples, n_features); got {X.ndim} dimension(s)."
        )
    if window < 1:
        raise ConfigurationError(f"window must be >= 1, got {window}.")
    if horizon < 1:
        raise ConfigurationError(f"horizon must be >= 1, got {horizon}.")

    n_samples = X.shape[0]
    n_windows = n_samples - window - horizon + 1
    if n_windows < 1:
        raise ConfigurationError(
            f"Not enough samples: need >= window + horizon = {window + horizon}, got {n_samples}."
        )

    # Gather indices for all windows at once: shape (n_windows, window).
    offsets = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    X3d = X[offsets]

    if y is None:
        return X3d, None

    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] != n_samples:
        raise ConfigurationError(
            f"y must have the same number of rows as X: {y.shape[0]} vs {n_samples}."
        )
    target_idx = np.arange(n_windows) + window - 1 + horizon
    y2d = y[target_idx].reshape(-1, 1)
    return X3d, y2d


class _BaseTorchRegressor(RegressorMixin, BaseEstimator):
    """Shared training/prediction machinery for the torch regressors.

    Subclasses set their hyper-parameters as attributes in ``__init__`` (so
    scikit-learn ``get_params`` / :func:`~sklearn.base.clone` work) and
    implement :meth:`_build_module`. Windowed subclasses additionally override
    :meth:`_fit_sequences` and :meth:`_predict_sequences`.
    """

    # Populated by subclass __init__; declared here for clarity only.
    epochs: int
    batch_size: int
    lr: float
    validation_fraction: float
    patience: int
    random_state: int

    # -------------------------------------------------------------- hooks
    def _build_module(self, n_features: int) -> Any:
        """Return the :class:`torch.nn.Module` for ``n_features`` inputs."""
        raise NotImplementedError

    def _fit_sequences(self, X_std: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map standardised ``(X, y)`` to model inputs/targets. Default: identity."""
        return X_std, y_std

    def _predict_sequences(self, torch: Any, module: Any, X_std: np.ndarray) -> np.ndarray:
        """Return standardised predictions of length ``len(X_std)``. Default: dense."""
        module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X_std.astype(np.float32), device=self.device_)
            out = module(tensor).cpu().numpy().ravel()
        return out.astype(np.float64)

    # --------------------------------------------------------------- API
    def fit(self, X: np.ndarray | Any, y: np.ndarray | Any) -> _BaseTorchRegressor:
        """Fit the network to ``X`` (2-D) and target ``y`` (1-D).

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target vector of length ``n_samples``.

        Returns
        -------
        self

        Raises
        ------
        ConfigurationError
            If ``X`` is not 2-D or its length does not match ``y``.
        nextaire_tools.exceptions.MissingDependencyError
            If PyTorch (the ``deep`` extra) is not installed.
        """
        torch = _torch()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ConfigurationError(
                f"X must be 2-D (n_samples, n_features); got {X.ndim} dimension(s)."
            )
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != X.shape[0]:
            raise ConfigurationError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

        self.n_features_in_ = X.shape[1]
        self._x_mean_ = X.mean(axis=0)
        self._x_std_ = X.std(axis=0)
        self._x_std_[self._x_std_ == 0.0] = 1.0
        self._y_mean_ = float(y.mean())
        y_std = float(y.std())
        self._y_std_ = y_std if y_std > 0.0 else 1.0

        X_std = (X - self._x_mean_) / self._x_std_
        y_scaled = (y - self._y_mean_) / self._y_std_
        inputs, targets = self._fit_sequences(X_std, y_scaled)
        if len(inputs) < 1:
            raise ConfigurationError("Not enough samples to build any training sequence.")

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self._train_module(torch, inputs, targets)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict targets for ``X``.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            1-D array of length ``n_samples``. For windowed models the first
            ``window`` rows are ``NaN`` (no full look-back window available).

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        torch = _torch()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ConfigurationError(
                f"X must be 2-D (n_samples, n_features); got {X.ndim} dimension(s)."
            )
        X_std = (X - self._x_mean_) / self._x_std_
        pred_std = self._predict_sequences(torch, self.module_, X_std)
        return pred_std * self._y_std_ + self._y_mean_

    # --------------------------------------------------------- internals
    def _train_module(self, torch: Any, inputs: np.ndarray, targets: np.ndarray) -> None:
        nn = torch.nn
        seed = int(self.random_state)
        torch.manual_seed(seed)
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(seed)
        device = self.device_

        module = self._build_module(self.n_features_in_).to(device)
        x_t = torch.as_tensor(np.asarray(inputs, dtype=np.float32), device=device)
        y_t = torch.as_tensor(np.asarray(targets, dtype=np.float32).reshape(-1, 1), device=device)

        n = x_t.shape[0]
        n_val = 0
        if self.validation_fraction and self.validation_fraction > 0.0:
            n_val = round(self.validation_fraction * n)
        use_val = 1 <= n_val < n
        if use_val:
            x_tr, y_tr = x_t[: n - n_val], y_t[: n - n_val]
            x_val, y_val = x_t[n - n_val :], y_t[n - n_val :]
        else:
            x_tr, y_tr = x_t, y_t
            x_val = y_val = None

        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        batch_size = max(1, int(self.batch_size))
        n_tr = x_tr.shape[0]

        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        no_improve = 0

        for epoch in range(int(self.epochs)):
            module.train()
            perm = torch.randperm(n_tr, generator=shuffle_gen)
            for start in range(0, n_tr, batch_size):
                idx = perm[start : start + batch_size].to(device)
                optimizer.zero_grad()
                loss = loss_fn(module(x_tr[idx]), y_tr[idx])
                loss.backward()
                optimizer.step()

            if x_val is not None:
                module.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(module(x_val), y_val).item())
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in module.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(self.patience):
                        _LOG.debug(
                            "Early stopping at epoch %d (best val loss %.6g).",
                            epoch,
                            best_val,
                        )
                        break

        if best_state is not None:
            module.load_state_dict(best_state)
        self.module_ = module

    def _check_is_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before 'predict'."
            )


class _WindowedTorchRegressor(_BaseTorchRegressor):
    """Base for sequence models that consume sliding windows of the series."""

    window: int

    def _fit_sequences(self, X_std: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_seq, y_seq = make_sequences(X_std, y_std, window=self.window, horizon=1)
        assert y_seq is not None  # y_std was provided
        return x_seq, y_seq.ravel()

    def _predict_sequences(self, torch: Any, module: Any, X_std: np.ndarray) -> np.ndarray:
        n = X_std.shape[0]
        out = np.full(n, np.nan, dtype=np.float64)
        if n < self.window + 1:
            # Not even one row has a full preceding window.
            return out
        x_seq, _ = make_sequences(X_std, None, window=self.window, horizon=1)
        module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x_seq.astype(np.float32), device=self.device_)
            preds = module(tensor).cpu().numpy().ravel()
        out[self.window : self.window + len(preds)] = preds
        return out


class MLPRegressor(_BaseTorchRegressor):
    """Feed-forward neural-network regressor (operates on 2-D features).

    Parameters
    ----------
    hidden_sizes:
        Widths of the hidden layers.
    dropout:
        Dropout probability applied after each hidden layer.
    epochs:
        Maximum number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    validation_fraction:
        Chronological tail fraction held out for early stopping. ``0`` disables
        the validation split (and early stopping).
    patience:
        Epochs without validation improvement before stopping early.
    random_state:
        Seed for deterministic weight initialisation and batching (CPU).

    Attributes
    ----------
    module_ : torch.nn.Module
        The trained network (available after :meth:`fit`).
    n_features_in_ : int
        Number of input features seen at fit time.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> X = np.random.default_rng(0).normal(size=(200, 3))  # doctest: +SKIP
    >>> y = X @ [1.0, -2.0, 0.5]  # doctest: +SKIP
    >>> model = MLPRegressor(epochs=50).fit(X, y)  # doctest: +SKIP
    >>> preds = model.predict(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        validation_fraction: float = 0.1,
        patience: int = 10,
        random_state: int = 0,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    def _build_module(self, n_features: int) -> Any:
        torch = _torch()
        nn = torch.nn
        layers: list[Any] = []
        in_dim = n_features
        for width in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, int(width)))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)


class LSTMRegressor(_WindowedTorchRegressor):
    """LSTM regressor over sliding time windows of a 2-D feature series.

    The input series is windowed internally via :func:`make_sequences`; the
    hidden state at the final time step feeds a linear read-out.
    :meth:`predict` returns one value per input row, with the first ``window``
    rows set to ``NaN``.

    Parameters
    ----------
    window:
        Look-back length (number of time steps per sequence).
    hidden_size:
        Number of LSTM hidden units.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability (between LSTM layers when ``num_layers > 1``, and
        before the read-out).
    epochs, batch_size, lr, validation_fraction, patience, random_state:
        Training controls; see :class:`MLPRegressor`.

    Attributes
    ----------
    module_ : torch.nn.Module
        The trained network (available after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> X = np.random.default_rng(0).normal(size=(300, 2))  # doctest: +SKIP
    >>> y = X[:, 0]  # doctest: +SKIP
    >>> model = LSTMRegressor(window=12, epochs=20).fit(X, y)  # doctest: +SKIP
    >>> np.isnan(model.predict(X)[:12]).all()  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        *,
        window: int = 24,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        validation_fraction: float = 0.1,
        patience: int = 10,
        random_state: int = 0,
    ) -> None:
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    def _build_module(self, n_features: int) -> Any:
        torch = _torch()
        nn = torch.nn
        hidden_size = int(self.hidden_size)
        num_layers = int(self.num_layers)
        dropout = float(self.dropout)

        class _LSTMModule(nn.Module):  # type: ignore[name-defined]  # dynamic torch base
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x: Any) -> Any:
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(self.dropout(last))

        return _LSTMModule()


class CNNRegressor(_WindowedTorchRegressor):
    """1-D convolutional regressor over sliding time windows of a 2-D series.

    Features are treated as input channels and the window as the temporal
    length; stacked ``Conv1d`` blocks are followed by global average pooling and
    a linear read-out. Shares the windowing / ``NaN``-padding prediction
    contract of :class:`LSTMRegressor`.

    Parameters
    ----------
    window:
        Look-back length (temporal extent of each window).
    channels:
        Output channels of each successive convolution block.
    kernel_size:
        Convolution kernel size (``'same'``-style padding preserves length).
    dropout:
        Dropout probability after each convolution block.
    epochs, batch_size, lr, validation_fraction, patience, random_state:
        Training controls; see :class:`MLPRegressor`.

    Attributes
    ----------
    module_ : torch.nn.Module
        The trained network (available after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> X = np.random.default_rng(0).normal(size=(300, 2))  # doctest: +SKIP
    >>> y = X[:, 0]  # doctest: +SKIP
    >>> model = CNNRegressor(window=12, epochs=20).fit(X, y)  # doctest: +SKIP
    >>> model.predict(X).shape[0] == 300  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        *,
        window: int = 24,
        channels: tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        validation_fraction: float = 0.1,
        patience: int = 10,
        random_state: int = 0,
    ) -> None:
        self.window = window
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    def _build_module(self, n_features: int) -> Any:
        torch = _torch()
        nn = torch.nn
        channels = tuple(int(c) for c in self.channels)
        kernel_size = int(self.kernel_size)
        dropout = float(self.dropout)
        padding = kernel_size // 2

        class _CNNModule(nn.Module):  # type: ignore[name-defined]  # dynamic torch base
            def __init__(self) -> None:
                super().__init__()
                blocks: list[Any] = []
                in_ch = n_features
                for out_ch in channels:
                    blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding))
                    blocks.append(nn.ReLU())
                    if dropout > 0.0:
                        blocks.append(nn.Dropout(dropout))
                    in_ch = out_ch
                self.conv = nn.Sequential(*blocks)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Linear(in_ch, 1)

            def forward(self, x: Any) -> Any:
                # (batch, window, features) -> (batch, features, window)
                x = x.transpose(1, 2)
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                return self.head(x)

        return _CNNModule()
