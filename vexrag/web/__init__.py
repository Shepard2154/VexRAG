"""Thin HTTP surface: import ``vexrag.core.attacks`` helpers here when wiring routes."""

from vexrag.core.attacks import (
    default_attack_registry,
    ensure_builtin_attacks_registered,
)

__all__ = [
    "default_attack_registry",
    "ensure_builtin_attacks_registered",
]
