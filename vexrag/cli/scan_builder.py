from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.core.attacks import (
    default_attack_registry,
    ensure_builtin_attacks_registered,
)
from vexrag.core.attacks.command import (
    ConfiguredScanCommand,
    ScanCommandProtocol,
)
from vexrag.core.scan_config_build import (
    build_corpus_poisoner,
    build_target_system,
)
from vexrag.core.scan_config_build import (
    build_evaluation_strategy as assemble_evaluation_strategy,
)

ScanCommand = ScanCommandProtocol


def build_scan_command(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    attack: str | None = None,
) -> ScanCommandProtocol:
    ensure_builtin_attacks_registered()
    registry = default_attack_registry()
    attack_id = (
        attack if attack is not None else registry.resolve_yaml_attack_key(config)
    )
    return registry.get(attack_id).build_scan_command(config, base_dir)


def resolve_attack_method(config: Mapping[str, Any]) -> str:
    ensure_builtin_attacks_registered()
    return default_attack_registry().resolve_yaml_attack_key(config)


def resolve_generate_cases_attack(
    config: Mapping[str, Any],
    *,
    explicit: str | None,
) -> str:
    ensure_builtin_attacks_registered()
    return default_attack_registry().resolve_generate_cases_attack(
        config,
        explicit=explicit,
    )


def build_evaluation_strategy(
    config: Mapping[str, Any],
    *,
    attack: str | None = None,
) -> Any:
    ensure_builtin_attacks_registered()
    registry = default_attack_registry()
    attack_id = (
        attack if attack is not None else registry.resolve_yaml_attack_key(config)
    )
    return assemble_evaluation_strategy(
        config,
        attack_id=attack_id,
        registry=registry,
    )


__all__ = [
    "ConfiguredScanCommand",
    "ScanCommand",
    "build_corpus_poisoner",
    "build_evaluation_strategy",
    "build_scan_command",
    "build_target_system",
    "resolve_attack_method",
    "resolve_generate_cases_attack",
]
