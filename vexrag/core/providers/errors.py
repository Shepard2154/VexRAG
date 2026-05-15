from vexrag.core.errors import ProviderServiceError

__all__ = ("ProviderConfigError", "ProviderServiceError")


class ProviderConfigError(ValueError):
    """Raised when provider configuration is missing or invalid."""
