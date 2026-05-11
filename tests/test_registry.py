"""Behavioural checks on ``AttackRegistry`` (stable across refactors)."""

from collections.abc import Mapping
from typing import Any

import pytest

from vexrag.core.attacks.plugin import AttackPlugin
from vexrag.core.attacks.registry import AttackRegistry, AttackRegistryError


def _noop_scan(config: Mapping[str, Any], base_dir: Any | None = None) -> Any:
    raise AssertionError("not used")


def _stub_plugin(attack_id: str = "stub_attack") -> AttackPlugin:
    return AttackPlugin(
        attack_id=attack_id,
        display_name="Stub",
        build_scan_command=_noop_scan,
        judge_prompt_builder_factory=None,
        build_automatic_case_generator=lambda c: object(),
        serialize_case_for_yaml=lambda x: {},
        generate_cases=lambda c, p: [],
    )


def test_registry_unknown_get_raises() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    with pytest.raises(AttackRegistryError, match="unknown attack"):
        reg.get("missing")
