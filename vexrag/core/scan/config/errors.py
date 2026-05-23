from vexrag.core.exceptions import ConfigurationError


class ScanConfigError(ConfigurationError):
    """Invalid scan configuration."""


class EvaluationConfigError(ScanConfigError):
    """Invalid evaluation configuration."""
