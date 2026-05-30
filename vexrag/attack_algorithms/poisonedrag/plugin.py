from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from vexrag.attack_algorithms.poison_base.case_id import stable_generated_case_id
from vexrag.attack_algorithms.poison_base.plugin_factory import (
    CorpusPoisonAttackSpec,
    build_attack_method_configurator,
)
from vexrag.attack_algorithms.poison_base.plugin_helpers import (
    build_attack_llm_client,
    load_attack_case_configs,
)
from vexrag.attack_algorithms.poisonedrag.case_generator import (
    AutomaticPoisonedRAGCaseGenerator,
)
from vexrag.attack_algorithms.poisonedrag.generator import PoisonedRAGGenerator
from vexrag.attack_algorithms.poisonedrag.prompts import PoisonedRAGJudgePromptBuilder
from vexrag.attack_algorithms.poisonedrag.scan_profile import (
    POISONEDRAG_ATTACK_ID,
    POISONEDRAG_SCAN_PROFILE,
)
from vexrag.attack_algorithms.poisonedrag.schema import PoisonedRAGRequest
from vexrag.core.attack_configurator import TargetStyle
from vexrag.core.attack_configurator.types import GenerateCasesParams
from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.builder import (
    attack_section,
    poisoning_style_option,
    target_style_option,
)
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    TargetCorrectAnswerProvider,
)


def build_poisonedrag_generator(
    config: Mapping[str, Any],
    target_system: HTTPTargetSystemAdapter,
    base_dir: Path | None,
    registries: ScanRegistries | None,
) -> PoisonedRAGGenerator:
    del base_dir
    attack_conf = attack_section(config, POISONEDRAG_ATTACK_ID)
    llm_client = build_attack_llm_client(
        config,
        attack_id=POISONEDRAG_ATTACK_ID,
        attack_config=attack_conf,
        registries=registries,
    )
    correct_answer_provider = None
    if str(attack_conf.get("correct_answer_provider", "")).strip() == "target_system":
        correct_answer_provider = TargetCorrectAnswerProvider(
            target_system=target_system,
            attack=POISONEDRAG_ATTACK_ID,
        )
    return PoisonedRAGGenerator(
        llm_client=llm_client,
        correct_answer_provider=correct_answer_provider,
    )


def build_poisonedrag_requests(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> tuple[PoisonedRAGRequest, ...]:
    attack_conf = attack_section(config, POISONEDRAG_ATTACK_ID)
    case_configs = load_attack_case_configs(
        attack_conf,
        attack_id=POISONEDRAG_ATTACK_ID,
        base_dir=base_dir,
    )
    if not case_configs:
        raise ScanConfigError(
            f"attack.{POISONEDRAG_ATTACK_ID}.cases or "
            f"attack.{POISONEDRAG_ATTACK_ID}.case_files "
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
    prefix = f"attack.{POISONEDRAG_ATTACK_ID}.cases[{case_number}]"
    case_config_accessor = ConfigAccessor(
        case_config,
        prefix=prefix,
        error_type=ScanConfigError,
    )
    attack_config_accessor = ConfigAccessor(
        attack_config,
        prefix=f"attack.{POISONEDRAG_ATTACK_ID}",
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
        adv_per_query=case_config_accessor.get_int(
            "adv_per_query", POISONEDRAG_SCAN_PROFILE.default_adv_per_query
        ),
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


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticPoisonedRAGCaseGenerator:
    attack_conf = attack_section(config, POISONEDRAG_ATTACK_ID)
    llm_client = build_attack_llm_client(
        config,
        attack_id=POISONEDRAG_ATTACK_ID,
        attack_config=attack_conf,
    )
    return AutomaticPoisonedRAGCaseGenerator(llm_client=llm_client)


def _serialize_case_for_yaml(case: Any) -> Mapping[str, Any]:
    case_id = str(getattr(case, "case_id", "") or "").strip()
    query = str(getattr(case, "query", "") or "").strip()
    correct_answer = str(getattr(case, "correct_answer", "") or "").strip()
    target_incorrect_answer = str(
        getattr(case, "target_incorrect_answer", "") or ""
    ).strip()
    if not query or not correct_answer or not target_incorrect_answer:
        raise ScanConfigError("generated case is missing required fields")
    if not case_id:
        case_id = stable_generated_case_id(query)
    adv_per_query = int(
        getattr(case, "adv_per_query", POISONEDRAG_SCAN_PROFILE.default_adv_per_query)
        or POISONEDRAG_SCAN_PROFILE.default_adv_per_query
    )
    return {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "target_incorrect_answer": target_incorrect_answer,
        "adv_per_query": adv_per_query,
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
            target_style=cast(TargetStyle, params.target_style),
            adv_per_query=params.adv_per_query,
            seed=params.seed,
        )
    )


POISONEDRAG_SPEC = CorpusPoisonAttackSpec(
    attack_id=POISONEDRAG_ATTACK_ID,
    display_name="PoisonedRAG",
    scan_profile=POISONEDRAG_SCAN_PROFILE,
    build_generator=build_poisonedrag_generator,
    build_requests=build_poisonedrag_requests,
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
    judge_prompt_builder_factory=lambda: PoisonedRAGJudgePromptBuilder(),
)

POISONEDRAG_PLUGIN = build_attack_method_configurator(POISONEDRAG_SPEC)
