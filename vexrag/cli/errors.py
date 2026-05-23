from vexrag.core.scan.config.errors import EvaluationConfigError, ScanConfigError


class CLIConfigError(ScanConfigError):
    """CLI-facing alias for invalid scan configuration (YAML or flags)."""


__all__ = ["CLIConfigError", "EvaluationConfigError", "ScanConfigError"]
