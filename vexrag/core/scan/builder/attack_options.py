from collections.abc import Mapping
from typing import Any

from vexrag.core.scan.config.errors import ScanConfigError


def attack_section(config: Mapping[str, Any], attack_name: str) -> Mapping[str, Any]:
    attack = config.get("attack")
    if not isinstance(attack, Mapping):
        raise ScanConfigError(f"attack.{attack_name} must be configured")
    attack_config = attack.get(attack_name)
    if not isinstance(attack_config, Mapping):
        raise ScanConfigError(f"attack.{attack_name} must be configured")
    return attack_config


def attack_llm_client_section(
    config: Mapping[str, Any],
    attack_config: Mapping[str, Any],
    *,
    attack: str,
) -> Mapping[str, Any]:
    client_config = (
        attack_config.get("llm_client")
        or attack_config.get("generator_client")
        or config.get("llm_client")
    )
    if not isinstance(client_config, Mapping):
        raise ScanConfigError(f"attack.{attack}.llm_client must be configured")
    return client_config
