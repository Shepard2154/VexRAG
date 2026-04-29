class CLIConfigError(ValueError):
    """Raised when CLI configuration is missing or invalid."""


class EvaluationConfigError(CLIConfigError):
    """Raised when CLI evaluation configuration is missing or invalid."""
