from dataclasses import dataclass

from vexrag.core.attack_configurator.errors import AttackMethodRegistryError
from vexrag.core.attack_configurator.types import AttackMethodConfigurator


@dataclass(frozen=True, slots=True)
class AttackMethodRegistry:
    """Immutable map of configured attack methods."""

    _by_id: dict[str, AttackMethodConfigurator]

    def get(self, attack_id: str) -> AttackMethodConfigurator:
        try:
            return self._by_id[attack_id.strip()]
        except KeyError as err:
            registered = ", ".join(sorted(self._by_id))
            raise AttackMethodRegistryError(
                f"unknown attack {attack_id!r}; registered attacks: {registered}"
            ) from err

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id.keys()))


class AttackMethodRegistryBuilder:
    """Mutable builder for an immutable AttackMethodRegistry."""

    def __init__(self) -> None:
        self._by_id: dict[str, AttackMethodConfigurator] = {}

    def register(self, method: AttackMethodConfigurator) -> None:
        attack_id = method.attack_id.strip()
        self._by_id[attack_id] = method

    def build(self) -> AttackMethodRegistry:
        return AttackMethodRegistry(dict(self._by_id))
