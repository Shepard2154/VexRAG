from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.core.attacks.registry import AttackRegistry, AttackRegistryError
from vexrag.core.config_errors import ScanConfigError
from vexrag.core.config_merge import deep_merge_mappings

_STEP_KEYS = frozenset({"id", "params", "scan", "evaluation", "evaluations"})


@dataclass(frozen=True, slots=True)
class AttackStepSpec:
    """One entry in the scan config ``attacks`` list."""
    attack_id: str
    params: Mapping[str, Any]
    scan_override: Mapping[str, Any] | None
    evaluation_override: Mapping[str, Any] | None
    evaluations_override: Mapping[str, Any] | None


def parse_attack_steps(
    config: Mapping[str, Any],
    registry: AttackRegistry,
) -> tuple[AttackStepSpec, ...]:
    """Parse and validate ``attacks`` from a scan YAML mapping."""
    if "attack" in config:
        raise ScanConfigError(
            "legacy top-level 'attack' is not supported; use a non-empty 'attacks' list"
        )
    raw = config.get("attacks")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ScanConfigError("attacks must be a non-empty list of attack steps")
    if not raw:
        raise ScanConfigError("attacks must be a non-empty list of attack steps")

    steps: list[AttackStepSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ScanConfigError(f"attacks[{index}] must be a mapping")
        unknown = frozenset(item) - _STEP_KEYS
        if unknown:
            raise ScanConfigError(
                f"attacks[{index}] has unknown keys: {', '.join(sorted(unknown))}"
            )
        attack_id_raw = item.get("id")
        if not isinstance(attack_id_raw, str) or not attack_id_raw.strip():
            raise ScanConfigError(f"attacks[{index}].id must be a non-empty string")
        attack_id = attack_id_raw.strip()
        registry.get(attack_id)

        params_raw = item.get("params", {})
        if not isinstance(params_raw, Mapping):
            raise ScanConfigError(f"attacks[{index}].params must be a mapping")

        scan_ov = item.get("scan")
        if scan_ov is not None and not isinstance(scan_ov, Mapping):
            raise ScanConfigError(f"attacks[{index}].scan must be a mapping when set")

        eval_ov = item.get("evaluation")
        if eval_ov is not None and not isinstance(eval_ov, Mapping):
            raise ScanConfigError(
                f"attacks[{index}].evaluation must be a mapping when set"
            )

        evals_ov = item.get("evaluations")
        if evals_ov is not None and not isinstance(evals_ov, Mapping):
            raise ScanConfigError(
                f"attacks[{index}].evaluations must be a mapping when set"
            )

        steps.append(
            AttackStepSpec(
                attack_id=attack_id,
                params=params_raw,
                scan_override=scan_ov,
                evaluation_override=eval_ov,
                evaluations_override=evals_ov,
            )
        )
    return tuple(steps)


def materialize_step_config(
    root: Mapping[str, Any],
    step: AttackStepSpec,
) -> dict[str, Any]:
    """Build a single-attack config dict (``attack.<id>``) for existing attack plugins."""
    out: dict[str, Any] = {
        key: value for key, value in root.items() if key not in ("attacks", "attack")
    }
    out["attack"] = {step.attack_id: dict(step.params)}

    base_scan = root.get("scan")
    if not isinstance(base_scan, Mapping):
        base_scan = {}
    if step.scan_override:
        out["scan"] = deep_merge_mappings(base_scan, step.scan_override)
    else:
        out["scan"] = dict(base_scan)

    base_eval = root.get("evaluation")
    if not isinstance(base_eval, Mapping):
        base_eval = {}
    if step.evaluation_override:
        out["evaluation"] = deep_merge_mappings(base_eval, step.evaluation_override)
    elif "evaluation" in root:
        out["evaluation"] = dict(base_eval)

    base_evaluations = root.get("evaluations")
    if not isinstance(base_evaluations, Mapping):
        base_evaluations = {}
    if step.evaluations_override:
        out["evaluations"] = deep_merge_mappings(
            base_evaluations,
            step.evaluations_override,
        )
    elif "evaluations" in root:
        out["evaluations"] = dict(base_evaluations)

    return out


def resolve_generate_cases_attack_id(
    config: Mapping[str, Any],
    registry: AttackRegistry,
    *,
    explicit: str | None,
) -> str:
    """Resolve which registered attack ``generate-cases`` should use."""
    steps = parse_attack_steps(config, registry)
    if explicit not in (None, "", "auto"):
        name = str(explicit).strip().lower()
        registry.get(name)
        if not any(s.attack_id == name for s in steps):
            raise AttackRegistryError(
                f"attacks list must include a step with id {name!r} for generate-cases"
            )
        return name

    if len(steps) != 1:
        raise AttackRegistryError(
            "multiple attack steps are configured; pass --attack <id> for generate-cases"
        )
    return steps[0].attack_id


def materialize_config_for_attack_id(
    root: Mapping[str, Any],
    attack_id: str,
    *,
    registry: AttackRegistry,
) -> dict[str, Any]:
    """Materialize the scan config for the first step matching ``attack_id``."""
    registry.get(attack_id)
    for step in parse_attack_steps(root, registry):
        if step.attack_id == attack_id:
            return materialize_step_config(root, step)
    raise AttackRegistryError(
        f"no attacks step with id {attack_id!r} in the scan YAML"
    )
