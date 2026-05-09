class ScanConfigError(ValueError):
    """Raised when scan YAML or programmatic scan assembly config is invalid."""


class EvaluationConfigError(ScanConfigError):
    """Raised when evaluation configuration is missing or invalid."""
