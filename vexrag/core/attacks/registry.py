from typing import Any

from vexrag.core.config_errors import ScanConfigError


class AttackRegistryError(ScanConfigError):
    """Raised when attack registry lookups fail."""


class AttackRegistry:
    """Registers built-in or third-party attack plugins by YAML ``attacks[].id``."""
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
