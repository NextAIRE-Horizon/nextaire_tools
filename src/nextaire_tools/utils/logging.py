"""Lightweight, opt-in logging for :mod:`nextaire_tools`.

The library follows the standard practice of never configuring the root logger on
import (it attaches a :class:`logging.NullHandler`).  Applications opt in with
:func:`enable_logging`.
"""

from __future__ import annotations

import logging

_LIBRARY_LOGGER_NAME = "nextaire_tools"

# Attach a NullHandler so "No handler found" warnings never appear for libraries
# that import nextaire_tools without configuring logging themselves.
logging.getLogger(_LIBRARY_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``nextaire_tools`` namespace.

    Parameters
    ----------
    name:
        Dotted suffix, typically ``__name__``. When ``None`` the root ``nextaire_tools``
        logger is returned.
    """
    if name is None or name == _LIBRARY_LOGGER_NAME:
        return logging.getLogger(_LIBRARY_LOGGER_NAME)
    if name.startswith(_LIBRARY_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LIBRARY_LOGGER_NAME}.{name}")


def enable_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Enable console logging for :mod:`nextaire_tools`.

    Convenience for interactive sessions and scripts. Idempotent: calling it
    repeatedly will not attach duplicate stream handlers.

    Parameters
    ----------
    level:
        Logging level (``logging.INFO``, ``"DEBUG"``, ...).

    Returns
    -------
    logging.Logger
        The configured ``nextaire_tools`` logger.
    """
    logger = logging.getLogger(_LIBRARY_LOGGER_NAME)
    logger.setLevel(level)
    already = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )
    if not already:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s"))
        logger.addHandler(handler)
    return logger


__all__ = ["get_logger", "enable_logging"]
