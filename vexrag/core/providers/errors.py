class ProviderConfigError(ValueError):
    """Raised when provider configuration is missing or invalid."""


class ProviderServiceError(RuntimeError):
    """Raised when a configured provider service is unavailable."""
