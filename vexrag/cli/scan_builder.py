from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.registries import (
    create_attack_method_registry,
    create_scan_registries,
)
from vexrag.core.scan.builder import build_evaluator as assemble_evaluator
from vexrag.core.scan.config import (
    ScanConfigError,
    materialize_config_for_attack_id,
    materialize_step_config,
    parse_attack_steps,
    resolve_generate_cases_attack_id,
)
from vexrag.core.scan.contracts import ScanCommand
from vexrag.core.scan.execution import (
    AttackChainScanCommand,
    probe_scan_llms_for_materialized_config,
)


def build_scan_command(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    attack: str | None = None,
) -> ScanCommand:
    attack_methods = create_attack_method_registry()
    registries = create_scan_registries(base_dir=base_dir)
    steps = parse_attack_steps(config, attack_methods)
    if attack is not None:
        aid = str(attack).strip().lower()
        attack_methods.get(aid)
        filtered = tuple(s for s in steps if s.attack_id == aid)
        if not filtered:
            raise ScanConfigError(f"no attack step with id {aid!r} in attacks list")
        steps = filtered
    if len(steps) == 1:
        materialized = materialize_step_config(config, steps[0])
        probe_scan_llms_for_materialized_config(
            materialized,
            attack_id=steps[0].attack_id,
            registries=registries,
            step_label=f"attack step ({steps[0].attack_id})",
        )
        return attack_methods.get(steps[0].attack_id).build_scan_command(
            materialized,
            base_dir,
        )
    built: list[tuple[str, ScanCommand]] = []
    for step_index, step in enumerate(steps, start=1):
        materialized = materialize_step_config(config, step)
        probe_scan_llms_for_materialized_config(
            materialized,
            attack_id=step.attack_id,
            registries=registries,
            step_label=f"chain step {step_index}/{len(steps)} ({step.attack_id})",
        )
        built.append(
            (
                step.attack_id,
                attack_methods.get(step.attack_id).build_scan_command(
                    materialized,
                    base_dir,
                ),
            )
        )
    return AttackChainScanCommand(tuple(built))


def resolve_attack_method(config: Mapping[str, Any]) -> str:
    attack_methods = create_attack_method_registry()
    steps = parse_attack_steps(config, attack_methods)
    return ",".join(step.attack_id for step in steps)


def resolve_generate_cases_attack(
    config: Mapping[str, Any],
    *,
    explicit: str | None,
) -> str:
    attack_methods = create_attack_method_registry()
    return resolve_generate_cases_attack_id(
        config,
        attack_methods,
        explicit=explicit,
    )


def build_evaluator(
    config: Mapping[str, Any],
    *,
    attack: str | None = None,
) -> Any:
    attack_methods = create_attack_method_registry()
    registries = create_scan_registries()
    steps = parse_attack_steps(config, attack_methods)
    if attack is not None:
        aid = str(attack).strip().lower()
        attack_methods.get(aid)
        selected = tuple(s for s in steps if s.attack_id == aid)
        if not selected:
            raise ScanConfigError(f"no attack step with id {aid!r} in attacks list")
        step = selected[0]
    elif len(steps) == 1:
        step = steps[0]
    else:
        raise ScanConfigError(
            "pass attack='...' to build_evaluator when multiple attacks are configured"
        )
    materialized = materialize_step_config(config, step)
    return assemble_evaluator(
        materialized,
        attack_id=step.attack_id,
        registries=registries,
    )


def materialize_generate_cases_config(
    config: Mapping[str, Any],
    *,
    attack_id: str,
) -> dict[str, Any]:
    """Return a single-attack config for ``generate_cases`` / plugin generators."""
    attack_methods = create_attack_method_registry()
    return materialize_config_for_attack_id(
        config,
        attack_id,
        registry=attack_methods,
    )
