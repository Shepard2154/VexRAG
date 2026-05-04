from enum import StrEnum


class ScanVerdict(StrEnum):
    """Final vulnerability verdict for a scan."""
    VULNERABLE = "vulnerable"
    NOT_VULNERABLE = "not_vulnerable"
