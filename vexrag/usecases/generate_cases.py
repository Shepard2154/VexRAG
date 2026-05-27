from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.registries import create_attack_method_registry
from vexrag.core.attack_configurator import GenerateCasesParams
from vexrag.usecases.config_io import write_yaml
from vexrag.usecases.errors import UseCaseConfigError
from vexrag.usecases.scan_service import (
    materialize_generate_cases_config,
    resolve_generate_cases_attack,
)


def default_adv_per_query(attack_id: str) -> int:
    return create_attack_method_registry().get(attack_id).default_adv_per_query


def run_generate_cases(
    config: Mapping[str, Any],
    *,
    attack: str,
    output: Path,
    count: int,
    topic: str | None,
    target_style: str,
    adv_per_query: int | None,
    seed: int | None,
    overwrite: bool,
    quiet: bool,
) -> Path:
    explicit = None if attack == "auto" else str(attack).strip().lower()
    attack_kind = resolve_generate_cases_attack(config, explicit=explicit)

    output_path = output.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if output_path.exists() and not overwrite:
        raise UseCaseConfigError(
            f"output file already exists: {output_path}. Use --overwrite to replace it."
        )

    registry = create_attack_method_registry()
    plugin = registry.get(attack_kind)
    if adv_per_query is None:
        adv = default_adv_per_query(attack_kind)
    else:
        adv = max(1, int(adv_per_query))

    params = GenerateCasesParams(
        count=int(count),
        topic=topic,
        target_style=str(target_style),
        seed=seed,
        adv_per_query=adv,
    )
    gen_config = materialize_generate_cases_config(
        config,
        attack_id=attack_kind,
    )
    cases = plugin.generate_cases(gen_config, params)
    payload = {"cases": [plugin.serialize_case_for_yaml(case) for case in cases]}
    write_yaml(output_path, payload)

    if quiet:
        print(output_path)
        return output_path

    yaml_attack = plugin.attack_id
    print(f"Generated {plugin.display_name} cases")
    print(f"Output: {output_path}")
    print(f"Cases: {len(cases)}")
    if topic:
        print(f"Topic: {topic}")
    print()
    print("Use in config:")
    print(
        "  attacks: [ { id: "
        f"{yaml_attack}, params: {{ case_files: ['{output_path}'], "
        f"adv_per_query: {adv} }} }} ]"
    )
    return output_path
