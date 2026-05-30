from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile
from vexrag.attack_algorithms.poison_base.runner import CorpusPoisonScanRunner
from vexrag.attack_algorithms.poison_base.scan_config import (
    build_corpus_poison_scan_config,
)
from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.attack_configurator.types import (
    AttackMethodConfigurator,
    GenerateCasesParams,
)
from vexrag.core.evaluation.contracts import JudgePromptBuilder
from vexrag.core.scan.builder import (
    build_corpus_poisoner,
    build_evaluator,
    build_target_system,
)
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.scan.execution import ConfiguredScanCommand
from vexrag.core.target_systems import HTTPTargetSystemAdapter


@dataclass(frozen=True, slots=True)
class CorpusPoisonAttackSpec:
    attack_id: str
    display_name: str
    scan_profile: CorpusPoisonScanProfile
    build_generator: Callable[
        [
            Mapping[str, Any],
            HTTPTargetSystemAdapter,
            Path | None,
            ScanRegistries | None,
        ],
        Any,
    ]
    build_requests: Callable[
        [Mapping[str, Any], Path | None],
        tuple[Any, ...],
    ]
    build_automatic_case_generator: Callable[[Mapping[str, Any]], Any]
    serialize_case_for_yaml: Callable[[Any], Mapping[str, Any]]
    generate_cases: Callable[
        [Mapping[str, Any], GenerateCasesParams],
        list[Any],
    ]
    judge_prompt_builder_factory: Callable[[], JudgePromptBuilder]


def build_corpus_poison_scan_command(
    spec: CorpusPoisonAttackSpec,
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> ConfiguredScanCommand:
    registries = create_scan_registries(base_dir=base_dir)
    target_system = build_target_system(
        config,
        registry=registries.target_systems,
    )
    if not isinstance(target_system, HTTPTargetSystemAdapter):
        raise ScanConfigError(
            f"{spec.attack_id} requires target_system.http configuration"
        )
    generator = spec.build_generator(
        config,
        target_system,
        base_dir,
        registries,
    )
    evaluator = build_evaluator(
        config,
        attack_id=spec.attack_id,
        registries=registries,
    )
    return ConfiguredScanCommand(
        runner=CorpusPoisonScanRunner(
            profile=spec.scan_profile,
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=build_corpus_poisoner(config, registries=registries),
        ),
        requests=spec.build_requests(config, base_dir),
        scan_config=build_corpus_poison_scan_config(config),
    )


def build_attack_method_configurator(
    spec: CorpusPoisonAttackSpec,
) -> AttackMethodConfigurator:
    def build_scan_command(
        config: Mapping[str, Any],
        base_dir: Path | None = None,
    ) -> ConfiguredScanCommand:
        return build_corpus_poison_scan_command(spec, config, base_dir=base_dir)

    return AttackMethodConfigurator(
        attack_id=spec.attack_id,
        display_name=spec.display_name,
        build_scan_command=build_scan_command,
        judge_prompt_builder_factory=spec.judge_prompt_builder_factory,
        build_automatic_case_generator=spec.build_automatic_case_generator,
        serialize_case_for_yaml=spec.serialize_case_for_yaml,
        generate_cases=spec.generate_cases,
        default_adv_per_query=spec.scan_profile.default_adv_per_query,
    )
