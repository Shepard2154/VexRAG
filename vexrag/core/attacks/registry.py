from collections.abc import Mapping
from typing import Any

from vexrag.core.config_errors import ScanConfigError


class AttackRegistryError(ScanConfigError):
    """Raised when attack registry lookups fail."""


class AttackRegistry:
    """Registers built-in or third-party attack plugins by YAML ``attack.<id>`` key."""
    __slots__ = ("_by_id",)

    def __init__(self) -> None:
        self._by_id: dict[str, Any] = {}

    def register(self, plugin: Any) -> None:
        attack_id = plugin.attack_id
        if not isinstance(attack_id, str) or not attack_id.strip():
            raise AttackRegistryError(
                "attack plugin attack_id must be a non-empty string"
            )
        normalized = attack_id.strip()
        if normalized in self._by_id:
            raise AttackRegistryError(f"attack '{normalized}' is already registered")
        self._by_id[normalized] = plugin

    def get(self, attack_id: str) -> Any:
        try:
            return self._by_id[attack_id.strip()]
        except KeyError as exc:
            supported = ", ".join(sorted(self._by_id))
            raise AttackRegistryError(
                f"unknown attack {attack_id!r}; supported attacks: {supported}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id.keys()))

    def resolve_yaml_attack_key(self, config: Mapping[str, Any]) -> str:
        """Return the single configured ``attack.<id>`` key present in ``config``."""
        attack = config.get("attack")
        if not isinstance(attack, Mapping):
            raise AttackRegistryError("attack must configure exactly one attack")

        candidates = [
            str(name).strip()
            for name, value in attack.items()
            if isinstance(value, Mapping) and isinstance(name, str) and name.strip()
        ]
        if len(candidates) != 1:
            raise AttackRegistryError("attack must configure exactly one attack")

        key = candidates[0]
        if key not in self._by_id:
            supported = ", ".join(sorted(self._by_id))
            raise AttackRegistryError(
                f"unknown attack {key!r} in YAML; supported attacks: {supported}"
            )
        return key

    def resolve_generate_cases_attack(
        self,
        config: Mapping[str, Any],
        *,
        explicit: str | None,
    ) -> str:
        """Resolve attack id for ``generate-cases`` (explicit flag or single YAML block)."""
        if explicit not in (None, "", "auto"):
            name = str(explicit).strip().lower()
            self.get(name)
            attack = config.get("attack")
            if not isinstance(attack, Mapping):
                raise AttackRegistryError("attack must be configured in the scan YAML")
            if name not in attack or not isinstance(attack[name], Mapping):
                raise AttackRegistryError(
                    f"attack.{name} must be configured in the scan YAML"
                )
            return name

        return self.resolve_yaml_attack_key(config)


_default_registry: AttackRegistry | None = None


def default_attack_registry() -> AttackRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = AttackRegistry()
    return _default_registry


def reset_default_attack_registry_for_tests() -> None:
    """Clear the process-wide registry (tests only)."""
    global _default_registry
    _default_registry = AttackRegistry()
