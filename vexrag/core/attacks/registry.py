from vexrag.core.attacks.plugin import AttackPlugin
from vexrag.core.config import ScanConfigError


class AttackRegistryError(ScanConfigError):
    """Raised when attack registry lookups fail."""


class AttackRegistry:
    """A plain in-memory map."""

    def __init__(self) -> None:
        self._by_id: dict[str, AttackPlugin] = {}

    def register(self, plugin: AttackPlugin) -> None:
        attack_id = plugin.attack_id.strip()
        self._by_id[attack_id] = plugin

    def get(self, attack_id: str) -> AttackPlugin:
        try:
            return self._by_id[attack_id.strip()]
        except KeyError as exc:
            registered = ", ".join(sorted(self._by_id))
            raise AttackRegistryError(
                f"unknown attack {attack_id!r}; registered attacks: {registered}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id.keys()))
