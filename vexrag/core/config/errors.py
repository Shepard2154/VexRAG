from vexrag.core.errors import ConfigError


class ScanConfigError(ConfigError):
    """Raised when scan YAML or programmatic scan assembly config is invalid."""


class EvaluationConfigError(ScanConfigError):
    """Raised when evaluation configuration is missing or invalid."""
