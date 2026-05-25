from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.core.llm import JSONGenerationLLMClientAdapter
from vexrag.core.llm.providers.defaults import create_default_llm_provider_registry
from vexrag.core.scan.builder import (
    attack_llm_client_section,
    attack_section,
    case_configs_from_value,
    load_case_configs,
    path_strings_from_value,
)
from vexrag.core.scan.builder.registries import ScanRegistries


def build_attack_llm_client(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    attack_config: Mapping[str, Any] | None = None,
    registries: ScanRegistries | None = None,
) -> JSONGenerationLLMClientAdapter:
    resolved_attack_config = attack_config or attack_section(config, attack_id)
    llm_client_config = attack_llm_client_section(
        config,
        resolved_attack_config,
        attack=attack_id,
    )
    providers = (
        registries.llm_providers
        if registries is not None
        else create_default_llm_provider_registry()
    )
    return JSONGenerationLLMClientAdapter(
        providers.build_json_completion_client(
            llm_client_config,
            config_prefix=f"attack.{attack_id}.llm_client",
        )
    )


def load_attack_case_configs(
    attack_config: Mapping[str, Any],
    *,
    attack_id: str,
    base_dir: Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    cases_prefix = f"attack.{attack_id}.cases"
    raw_cases = attack_config.get("cases", ())
    inline = case_configs_from_value(raw_cases, cases_prefix)

    file_paths = [
        *path_strings_from_value(
            attack_config.get("case_files", ()),
            f"attack.{attack_id}.case_files",
        ),
        # TODO: Is it really necessary?
        *path_strings_from_value(
            attack_config.get("cases_file", ()),
            f"attack.{attack_id}.cases_file",
        ),
    ]
    from_files: list[Mapping[str, Any]] = []
    for file_path in file_paths:
        from_files.extend(load_case_configs(file_path, base_dir=base_dir))
    return (*inline, *from_files)
