from vexrag.core.exceptions import DependencyServiceError


class TargetSystemAdapterError(DependencyServiceError):
    """Raised when a target system cannot be queried or decoded."""
