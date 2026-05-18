from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.poisonedrag.case_generator import (
    AutomaticPoisonedRAGCaseGenerator,
)
from vexrag.attack_algorithms.poisonedrag.evaluation import (
    PoisonedRAGJudgePromptBuilder,
)
from vexrag.attack_algorithms.poisonedrag.generator import PoisonedRAGGenerator
from vexrag.attack_algorithms.poisonedrag.scan import (
    PoisonedRAGScanConfig,
    PoisonedRAGScanRunner,
)
from vexrag.attack_algorithms.poisonedrag.schema import PoisonedRAGRequest
from vexrag.core.attacks.command import ConfiguredScanCommand
from vexrag.core.attacks.plugin import AttackPlugin, GenerateCasesParams
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import ScanConfigAccessor, ScanConfigError
from vexrag.core.config.build import (
    attack_llm_client_section,
    attack_section,
    build_evaluator,
    build_retrieval_corpus_adapter,
    build_target_system,
    case_configs_from_value,
    cleanup_option,
    load_case_configs,
    path_strings_from_value,
    poisoning_style_option,
    target_style_option,
)
from vexrag.core.json_generation_client import JSONGenerationLLMClientAdapter
from vexrag.core.providers import build_judge_client as build_provider_judge_client
from vexrag.core.target import HTTPTargetSystemAdapter
from vexrag.core.target_correct_answer import TargetCorrectAnswerProvider


def _build_poison_llm_client(
    config: Mapping[str, Any],
    *,
    attack_config: Mapping[str, Any] | None = None,
) -> JSONGenerationLLMClientAdapter:
    resolved_attack_config = attack_config or attack_section(config, "poisonedrag")
    llm_client_config = attack_llm_client_section(
        config,
        resolved_attack_config,
        attack="poisonedrag",
    )
    return JSONGenerationLLMClientAdapter(
        build_provider_judge_client(llm_client_config)
    )


def build_poisonedrag_generator(
    config: Mapping[str, Any],
    *,
    target_system: HTTPTargetSystemAdapter,
) -> PoisonedRAGGenerator:
    attack_conf = attack_section(config, "poisonedrag")
    llm_client = _build_poison_llm_client(config, attack_config=attack_conf)
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
    case_configs = [
        *_inline_case_configs_poison(attack_conf),
        *_case_file_configs_poison(attack_conf, base_dir=base_dir),
    ]
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
    case_config_accessor = ScanConfigAccessor(case_config, prefix=prefix)
    attack_config_accessor = ScanConfigAccessor(
        attack_config, prefix="attack.poisonedrag"
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


def _inline_case_configs_poison(
    attack_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_cases = attack_config.get("cases", ())
    return case_configs_from_value(raw_cases, "attack.poisonedrag.cases")


def _case_file_configs_poison(
    attack_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    file_paths = [
        *path_strings_from_value(
            attack_config.get("case_files", ()),
            "attack.poisonedrag.case_files",
        ),
        *path_strings_from_value(
            attack_config.get("cases_file", ()),
            "attack.poisonedrag.cases_file",
        ),
    ]
    cases: list[Mapping[str, Any]] = []
    for file_path in file_paths:
        cases.extend(load_case_configs(file_path, base_dir=base_dir))
    return tuple(cases)


def build_poisonedrag_scan_config(config: Mapping[str, Any]) -> PoisonedRAGScanConfig:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise ScanConfigError("scan must be a mapping")
    scan_config_accessor = ScanConfigAccessor(scan_config, prefix="scan")
    return PoisonedRAGScanConfig(
        repetitions=scan_config_accessor.get_optional_int("repetitions", 1),
        attack_success_rate_threshold=scan_config_accessor.get_optional_float(
            "attack_success_rate_threshold", 0.0
        ),
        override_contexts=scan_config_accessor.get_bool("override_contexts", False),
        cleanup=cleanup_option(scan_config),
    )


def _build_scan_command(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> ConfiguredScanCommand:
    registry = AttackRegistry()
    registry.register(POISON_PLUGIN)
    target_system = build_target_system(config)
    generator = build_poisonedrag_generator(config, target_system=target_system)
    evaluator = build_evaluator(
        config,
        attack_id="poisonedrag",
        registry=registry,
    )
    return ConfiguredScanCommand(
        runner=PoisonedRAGScanRunner(
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_adapter=build_retrieval_corpus_adapter(config, base_dir=base_dir),
        ),
        requests=build_poisonedrag_requests(config, base_dir=base_dir),
        scan_config=build_poisonedrag_scan_config(config),
    )


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticPoisonedRAGCaseGenerator:
    attack_conf = attack_section(config, "poisonedrag")
    llm_client = _build_poison_llm_client(config, attack_config=attack_conf)
    return AutomaticPoisonedRAGCaseGenerator(llm_client=llm_client)


def _serialize_case_for_yaml(case: Any) -> Mapping[str, str]:
    case_id = str(getattr(case, "case_id", "") or "").strip()
    query = str(getattr(case, "query", "")).strip()
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


POISON_PLUGIN = AttackPlugin(
    attack_id="poisonedrag",
    display_name="PoisonedRAG",
    build_scan_command=_build_scan_command,
    judge_prompt_builder_factory=lambda: PoisonedRAGJudgePromptBuilder(),
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
)
