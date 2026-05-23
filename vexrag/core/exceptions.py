class VexRAGCoreError(Exception):
    """Base class for core package errors."""


class ConfigurationError(ValueError, VexRAGCoreError):
    """Base class for configuration validation errors."""


class DependencyServiceError(RuntimeError, VexRAGCoreError):
    """Base class for external dependency service errors."""
