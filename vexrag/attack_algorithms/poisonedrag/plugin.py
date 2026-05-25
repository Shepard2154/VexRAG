from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.poison_base.plugin_helpers import (
    build_attack_llm_client,
    load_attack_case_configs,
)
from vexrag.attack_algorithms.poison_base.scan_config import (
    build_corpus_poison_scan_config,
)
from vexrag.attack_algorithms.poisonedrag.case_generator import (
    AutomaticPoisonedRAGCaseGenerator,
)
from vexrag.attack_algorithms.poisonedrag.evaluation import (
    PoisonedRAGJudgePromptBuilder,
)
from vexrag.attack_algorithms.poisonedrag.generator import PoisonedRAGGenerator
from vexrag.attack_algorithms.poisonedrag.scan import PoisonedRAGScanRunner
from vexrag.attack_algorithms.poisonedrag.schema import PoisonedRAGRequest
from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.attack_configurator.types import (
    AttackMethodConfigurator,
    GenerateCasesParams,
)
from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.builder import (
    attack_section,
    build_corpus_poisoner,
    build_evaluator,
    build_target_system,
    poisoning_style_option,
    target_style_option,
)
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.scan.execution import ConfiguredScanCommand
from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    TargetCorrectAnswerProvider,
)


def build_poisonedrag_generator(
    config: Mapping[str, Any],
    *,
    target_system: HTTPTargetSystemAdapter,
    registries: ScanRegistries | None = None,
) -> PoisonedRAGGenerator:
    attack_conf = attack_section(config, "poisonedrag")
    llm_client = build_attack_llm_client(
        config,
        attack_id="poisonedrag",
        attack_config=attack_conf,
        registries=registries,
    )
    correct_answer_provider = None
    if str(attack_conf.get("correct_answer_provider", "")).strip() == "target_system":
        correct_answer_provider = TargetCorrectAnswerProvider(
            target_system=target_system,
            attack="poisonedrag",
        )
    return PoisonedRAGGenerator(
        llm_client=llm_client,
        correct_answer_provider=correct_answer_provider,
    )


def build_poisonedrag_requests(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> tuple[PoisonedRAGRequest, ...]:
    attack_conf = attack_section(config, "poisonedrag")
    case_configs = load_attack_case_configs(
        attack_conf,
        attack_id="poisonedrag",
        base_dir=base_dir,
    )
    if not case_configs:
        raise ScanConfigError(
            "attack.poisonedrag.cases or attack.poisonedrag.case_files "
            "must contain at least one case"
        )
    return tuple(
        _build_poisonedrag_request(
            attack_conf,
            case_config,
            case_number=case_number,
        )
        for case_number, case_config in enumerate(case_configs, start=1)
    )


def _build_poisonedrag_request(
    attack_config: Mapping[str, Any],
    case_config: Mapping[str, Any],
    *,
    case_number: int,
) -> PoisonedRAGRequest:
    prefix = f"attack.poisonedrag.cases[{case_number}]"
    case_config_accessor = ConfigAccessor(
        case_config,
        prefix=prefix,
        error_type=ScanConfigError,
    )
    attack_config_accessor = ConfigAccessor(
        attack_config,
        prefix="attack.poisonedrag",
        error_type=ScanConfigError,
    )
    return PoisonedRAGRequest(
        query=case_config_accessor.get_required_string("query"),
        correct_answer=case_config_accessor.get_optional_string("correct_answer"),
        target_incorrect_answer=case_config_accessor.get_optional_string(
            "target_incorrect_answer"
        ),
        case_id=case_config_accessor.get_optional_string(
            "case_id", case_config_accessor.get_optional_string("id")
        ),
        adv_per_query=case_config_accessor.get_optional_int("adv_per_query", 3),
        target_style=target_style_option(
            case_config,
            default=target_style_option(attack_config),
            prefix=prefix,
        ),
        poisoning_style=poisoning_style_option(
            case_config,
            default=poisoning_style_option(attack_config),
            prefix=prefix,
        ),
        seed=case_config_accessor.get_optional_int(
            "seed", attack_config_accessor.get_optional_int("seed")
        ),
    )


def _build_scan_command(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> ConfiguredScanCommand:
    registries = create_scan_registries(base_dir=base_dir)
    target_system = build_target_system(
        config,
        registry=registries.target_systems,
    )
    generator = build_poisonedrag_generator(
        config,
        target_system=target_system,
        registries=registries,
    )
    evaluator = build_evaluator(
        config,
        attack_id="poisonedrag",
        registries=registries,
    )
    return ConfiguredScanCommand(
        runner=PoisonedRAGScanRunner(
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=build_corpus_poisoner(config, registries=registries),
        ),
        requests=build_poisonedrag_requests(config, base_dir=base_dir),
        scan_config=build_corpus_poison_scan_config(config),
    )


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticPoisonedRAGCaseGenerator:
    attack_conf = attack_section(config, "poisonedrag")
    llm_client = build_attack_llm_client(
        config,
        attack_id="poisonedrag",
        attack_config=attack_conf,
    )
    return AutomaticPoisonedRAGCaseGenerator(llm_client=llm_client)


def _serialize_case_for_yaml(case: Any) -> Mapping[str, str]:
    case_id = str(getattr(case, "case_id", "") or "").strip()
    query = str(getattr(case, "query", "") or "").strip()
    correct_answer = str(getattr(case, "correct_answer", "") or "").strip()
    target_incorrect_answer = str(
        getattr(case, "target_incorrect_answer", "") or ""
    ).strip()
    if not query or not correct_answer or not target_incorrect_answer:
        raise ScanConfigError("generated case is missing required fields")
    if not case_id:
        case_id = f"generated_case_{abs(hash(query)) % 1_000_000}"
    return {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "target_incorrect_answer": target_incorrect_answer,
    }


def _generate_cases(
    config: Mapping[str, Any],
    params: GenerateCasesParams,
) -> list[Any]:
    generator = _build_automatic_case_generator(config)
    return list(
        generator.generate_cases(
            count=params.count,
            topic=params.topic,
            target_style=params.target_style,
            seed=params.seed,
        )
    )


POISON_PLUGIN = AttackMethodConfigurator(
    attack_id="poisonedrag",
    display_name="PoisonedRAG",
    build_scan_command=_build_scan_command,
    judge_prompt_builder_factory=lambda: PoisonedRAGJudgePromptBuilder(),
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
)
