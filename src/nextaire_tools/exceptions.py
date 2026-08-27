"""Exception hierarchy for :mod:`nextaire_tools`.

All errors raised by the library derive from :class:`NextaireToolsError`, so callers can
catch every library-specific failure with a single ``except NextaireToolsError``.
"""

from __future__ import annotations


class NextaireToolsError(Exception):
    """Base class for every exception raised by :mod:`nextaire_tools`."""


class NotFittedError(NextaireToolsError):
    """Raised when ``transform`` is called before ``fit`` on a step."""


class ColumnNotFoundError(NextaireToolsError, KeyError):
    """Raised when a requested column is absent from the input frame."""


class SchemaError(NextaireToolsError, ValueError):
    """Raised when the input data does not match the expected schema."""


class ConfigurationError(NextaireToolsError, ValueError):
    """Raised for invalid step / model configuration."""


class MissingDependencyError(NextaireToolsError, ImportError):
    """Raised when an optional dependency is required but not installed.

    Parameters
    ----------
    package:
        The importable module name that failed to import.
    extra:
        The pip *extra* that provides the dependency (e.g. ``"deep"``), used to
        build an actionable installation hint.
    feature:
        Human-readable description of the feature that needs the dependency.
    """

    def __init__(self, package: str, extra: str, feature: str = "") -> None:
        feature_txt = f" for {feature}" if feature else ""
        message = (
            f"The optional dependency '{package}' is required{feature_txt} but is not "
            f"installed.\nInstall it with:  pip install 'nextaire_tools[{extra}]'"
        )
        super().__init__(message)
        self.package = package
        self.extra = extra
        self.feature = feature


class ExtractionError(NextaireToolsError):
    """Raised when a Copernicus / remote data extraction fails."""


class CredentialsError(ExtractionError):
    """Raised when Copernicus Data Store credentials are missing or invalid."""
