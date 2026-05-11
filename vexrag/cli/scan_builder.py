from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN
from vexrag.core.attack_plan import (
    materialize_config_for_attack_id,
    materialize_step_config,
    parse_attack_steps,
    resolve_generate_cases_attack_id,
)
from vexrag.core.attacks.chain_command import AttackChainScanCommand
from vexrag.core.attacks.command import (
    ScanCommandProtocol,
)
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import ScanConfigError
from vexrag.core.config.build import (
    build_evaluation_strategy as assemble_evaluation_strategy,
)
from vexrag.core.llm_scan_probe import probe_scan_llms_for_materialized_config

ScanCommand = ScanCommandProtocol


def build_scan_command(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    attack: str | None = None,
) -> ScanCommandProtocol:
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    steps = parse_attack_steps(config, registry)
    if attack is not None:
        aid = str(attack).strip().lower()
        registry.get(aid)
        filtered = tuple(s for s in steps if s.attack_id == aid)
        if not filtered:
            raise ScanConfigError(f"no attack step with id {aid!r} in attacks list")
        steps = filtered
    if len(steps) == 1:
        materialized = materialize_step_config(config, steps[0])
        probe_scan_llms_for_materialized_config(
            materialized,
            attack_id=steps[0].attack_id,
            step_label=f"attack step ({steps[0].attack_id})",
        )
        return registry.get(steps[0].attack_id).build_scan_command(
            materialized,
            base_dir,
        )
    built: list[tuple[str, ScanCommandProtocol]] = []
    for step_index, step in enumerate(steps, start=1):
        materialized = materialize_step_config(config, step)
        probe_scan_llms_for_materialized_config(
            materialized,
            attack_id=step.attack_id,
            step_label=f"chain step {step_index}/{len(steps)} ({step.attack_id})",
        )
        built.append(
            (
                step.attack_id,
                registry.get(step.attack_id).build_scan_command(
                    materialized,
                    base_dir,
                ),
            )
        )
    return AttackChainScanCommand(tuple(built))


def resolve_attack_method(config: Mapping[str, Any]) -> str:
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    steps = parse_attack_steps(config, registry)
    return ",".join(step.attack_id for step in steps)


def resolve_generate_cases_attack(
    config: Mapping[str, Any],
    *,
    explicit: str | None,
) -> str:
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    return resolve_generate_cases_attack_id(config, registry, explicit=explicit)


def build_evaluation_strategy(
    config: Mapping[str, Any],
    *,
    attack: str | None = None,
) -> Any:
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    steps = parse_attack_steps(config, registry)
    if attack is not None:
        aid = str(attack).strip().lower()
        registry.get(aid)
        selected = tuple(s for s in steps if s.attack_id == aid)
        if not selected:
            raise ScanConfigError(f"no attack step with id {aid!r} in attacks list")
        step = selected[0]
    elif len(steps) == 1:
        step = steps[0]
    else:
        raise ScanConfigError(
            "pass attack='...' to build_evaluation_strategy when multiple attacks "
            "are configured"
        )
    materialized = materialize_step_config(config, step)
    return assemble_evaluation_strategy(
        materialized,
        attack_id=step.attack_id,
        registry=registry,
    )


def materialize_generate_cases_config(
    config: Mapping[str, Any],
    *,
    attack_id: str,
) -> dict[str, Any]:
    """Return a single-attack config for ``generate_cases`` / plugin generators."""
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    return materialize_config_for_attack_id(config, attack_id, registry=registry)
