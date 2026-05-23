from vexrag.core.exceptions import ConfigurationError, DependencyServiceError


class ProviderConfigError(ConfigurationError):
    """Raised when provider configuration is missing or invalid."""


class ProviderServiceError(DependencyServiceError):
    """Raised when a configured provider service is unavailable."""
