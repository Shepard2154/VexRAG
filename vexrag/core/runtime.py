"""Programmatic API: isolated registry and idempotent builtin registration.

Use :func:`create_runtime` so each integration gets its own
:class:`AttackRegistry` and builtin registration flag. The process-wide
``default_attack_registry`` (see ``vexrag.core.attacks.registry``) is for CLI
convenience and shares mutable state across callers in the same process.
"""

from dataclasses import dataclass, field

from vexrag.core.attacks.builtins import (
    load_attack_entry_points,
    register_builtin_attacks,
)
from vexrag.core.attacks.registry import AttackRegistry


@dataclass
class VexRAGRuntime:
    """Bundles an ``AttackRegistry`` with one-time builtin and entry-point loading.

    Recommended for services and libraries: keep one instance (or one per
    isolation boundary) instead of using the global default registry from
    :func:`~vexrag.core.attacks.registry.default_attack_registry`.
    """

    registry: AttackRegistry = field(default_factory=AttackRegistry)
    _builtins_loaded: bool = field(default=False, repr=False)

    def ensure_builtin_attacks_registered(self) -> AttackRegistry:
        if self._builtins_loaded:
            return self.registry
        register_builtin_attacks(self.registry)
        load_attack_entry_points(self.registry)
        self._builtins_loaded = True
        return self.registry

    def reset_builtin_registration_for_tests(self) -> None:
        self._builtins_loaded = False


def create_runtime() -> VexRAGRuntime:
    """Return a new runtime with its own registry and builtin state.

    This is the supported embedding path when you must not share attack
    registration with other code in the same process (tests, other requests, or
    workers). It does not use ``default_attack_registry``.
    """
    return VexRAGRuntime()
