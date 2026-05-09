"""Scan configuration: errors, merging shallow config trees, and typed YAML option parsing."""

from .errors import EvaluationConfigError, ScanConfigError
from .merge import deep_merge_mappings

__all__ = [
    "EvaluationConfigError",
    "ScanConfigError",
    "deep_merge_mappings",
]
