from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.builder import cleanup_option
from vexrag.core.scan.config.errors import ScanConfigError


@dataclass(frozen=True, slots=True)
class CorpusPoisonScanConfig:
    """Execution settings for a corpus-poison target-system scan."""

    repetitions: int = 1
    attack_success_rate_threshold: float = 0.0
    override_contexts: bool = False
    cleanup: bool = False

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0 <= self.attack_success_rate_threshold <= 1:
            raise ValueError("attack_success_rate_threshold must be between 0 and 1")


def build_corpus_poison_scan_config(
    config: Mapping[str, Any],
) -> CorpusPoisonScanConfig:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise ScanConfigError("scan must be a mapping")
    scan_config_accessor = ConfigAccessor(
        scan_config,
        prefix="scan",
        error_type=ScanConfigError,
    )
    return CorpusPoisonScanConfig(
        repetitions=scan_config_accessor.get_optional_int("repetitions", 1),
        attack_success_rate_threshold=scan_config_accessor.get_optional_float(
            "attack_success_rate_threshold", 0.0
        ),
        override_contexts=scan_config_accessor.get_bool("override_contexts", False),
        cleanup=cleanup_option(scan_config),
    )
