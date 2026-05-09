"""Attack registry storage and optional process-wide default.

:class:`AttackRegistry` is a plain in-memory map; it is not thread-safe for
concurrent mutation.

:func:`default_attack_registry` exposes a lazily created process-wide singleton
for CLI and one-off scripts. That default is **not** synchronized: two threads
could theoretically race on first initialization, and concurrent
``register``/``get`` without external locking is unsupported.

When embedding VexRAG in a service, **prefer**
:func:`vexrag.core.runtime.create_runtime` and use only that
:class:`~vexrag.core.runtime.VexRAGRuntime` instance (call
:meth:`~vexrag.core.runtime.VexRAGRuntime.ensure_builtin_attacks_registered`).
That avoids hidden coupling when multiple tests, worker threads, or async tasks
in the same process would otherwise share one default registry.

Separate OS processes (for example pytest-xdist workers) each have their own
default; coupling is mainly within a single process.

:func:`reset_default_attack_registry_for_tests` replaces the process-wide
default for tests only.
"""

from vexrag.core.attacks.plugin import AttackPlugin
from vexrag.core.config import ScanConfigError


class AttackRegistryError(ScanConfigError):
    """Raised when attack registry lookups fail."""


class AttackRegistry:
    """Registers built-in or third-party attack plugins by YAML ``attacks[].id``."""
    __slots__ = ("_by_id",)

    def __init__(self) -> None:
        self._by_id: dict[str, AttackPlugin] = {}

    def register(self, plugin: AttackPlugin) -> None:
        attack_id = plugin.attack_id
        if not isinstance(attack_id, str) or not attack_id.strip():
            raise AttackRegistryError(
                "attack plugin attack_id must be a non-empty string"
            )
        normalized = attack_id.strip()
        if normalized in self._by_id:
            raise AttackRegistryError(f"attack '{normalized}' is already registered")
        self._by_id[normalized] = plugin

    def get(self, attack_id: str) -> AttackPlugin:
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
    """Return the process-wide default registry (lazy singleton).

    Intended for CLI and simple entrypoints. For isolated state in applications,
    use :func:`vexrag.core.runtime.create_runtime` instead; see the module
    docstring for threading and embedding guidance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = AttackRegistry()
    return _default_registry


def reset_default_attack_registry_for_tests() -> None:
    """Replace the process-wide registry with a fresh instance (tests only).

    Affects only the current process; it does not reset registries in other
    worker processes.
    """
    global _default_registry
    _default_registry = AttackRegistry()
