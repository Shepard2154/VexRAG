"""Import attack packages once so each ``plugin`` module registers with the default registry."""

import logging

from vexrag.core.attacks.registry import AttackRegistry, default_attack_registry

LOGGER = logging.getLogger("vexrag.attacks.builtins")


def register_builtin_attacks(registry: AttackRegistry | None = None) -> AttackRegistry:
    """Register all attacks shipped with VexRAG. Safe to call multiple times."""
    reg = registry if registry is not None else default_attack_registry()
    from vexrag.attack_algorithms.hijackrag import (
        plugin as hijack_plugin,  # noqa: PLC0415
    )
    from vexrag.attack_algorithms.poisonedrag import (
        plugin as poison_plugin,  # noqa: PLC0415
    )

    hijack_plugin.register(reg)
    poison_plugin.register(reg)
    return reg


def _load_attack_entry_points(registry: AttackRegistry) -> None:
    try:
        import importlib.metadata as im
    except ImportError:
        return

    eps_collection = im.entry_points()
    if hasattr(eps_collection, "select"):
        eps = eps_collection.select(group="vexrag.attacks")
    else:
        eps = eps_collection.get("vexrag.attacks", ())

    for ep in eps:
        try:
            register_fn = ep.load()
        except Exception as exc:
            LOGGER.warning(
                "skipped broken vexrag.attacks entry point %s: %s", ep.name, exc
            )
            continue
        try:
            register_fn(registry)
        except Exception as exc:
            LOGGER.warning(
                "entry point %s failed during registration: %s", ep.name, exc
            )


_builtins_registered = False


def ensure_builtin_attacks_registered() -> AttackRegistry:
    """Idempotent registration for CLI/web entrypoints."""
    global _builtins_registered
    reg = default_attack_registry()
    if _builtins_registered:
        return reg
    register_builtin_attacks(reg)
    _load_attack_entry_points(reg)
    _builtins_registered = True
    return reg


def reset_builtin_registration_for_tests() -> None:
    global _builtins_registered
    _builtins_registered = False
