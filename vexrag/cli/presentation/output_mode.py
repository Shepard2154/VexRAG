from enum import Enum


class OutputMode(Enum):
    QUIET = "quiet"
    COMPACT = "compact"
    DETAILED = "detailed"


def resolve_output_mode(*, quiet: bool, detailed: bool) -> OutputMode:
    if quiet:
        return OutputMode.QUIET
    if detailed:
        return OutputMode.DETAILED
    return OutputMode.COMPACT
