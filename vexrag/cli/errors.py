from vexrag.core.config_errors import EvaluationConfigError, ScanConfigError


class CLIConfigError(ScanConfigError):
    """CLI-facing alias for invalid scan configuration (YAML or flags)."""


__all__ = ["CLIConfigError", "EvaluationConfigError"]
