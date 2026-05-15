from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.hijackrag.case_generator import (
    AutomaticHijackRAGCaseGenerator,
)
from vexrag.attack_algorithms.hijackrag.evaluation import HijackRAGJudgePromptBuilder
from vexrag.attack_algorithms.hijackrag.generator import HijackRAGGenerator
from vexrag.attack_algorithms.hijackrag.scan import (
    HijackRAGScanConfig,
    HijackRAGScanRunner,
)
from vexrag.attack_algorithms.hijackrag.schema import HijackRAGRequest
from vexrag.attack_algorithms.hijackrag.segments import default_hijack_segments_path
from vexrag.core.attacks.command import ConfiguredScanCommand
from vexrag.core.attacks.plugin import AttackPlugin, GenerateCasesParams
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import ScanConfigError
from vexrag.core.config.build import (
    attack_llm_client_section,
    attack_section,
    build_corpus_poisoner,
    build_evaluator,
    build_target_system,
    case_configs_from_value,
    cleanup_option,
    correct_answer_style_option,
    load_case_configs,
    path_strings_from_value,
)
from vexrag.core.config.options import ConfigAccessor
from vexrag.core.json_generation_client import JSONGenerationLLMClientAdapter
from vexrag.core.providers import build_judge_client as build_provider_judge_client
from vexrag.core.target import HTTPTargetSystemAdapter
from vexrag.core.target_correct_answer import TargetCorrectAnswerProvider


def _hijack_segments_path(
    attack_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> Path:
    raw = attack_config.get("segments_file")
    if raw in (None, ""):
        return default_hijack_segments_path()
    if not isinstance(raw, str) or not raw.strip():
        raise ScanConfigError(
            "attack.hijackrag.segments_file must be a non-empty string when set"
        )
    path = Path(raw.strip())
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ScanConfigError(f"attack.hijackrag.segments_file not found: {path}")
    return path


def build_hijackrag_llm_client(
    config: Mapping[str, Any],
    *,
    attack_config: Mapping[str, Any] | None = None,
) -> JSONGenerationLLMClientAdapter:
    resolved = attack_config or attack_section(config, "hijackrag")
    llm_client_config = attack_llm_client_section(
        config,
        resolved,
        attack="hijackrag",
    )
    return JSONGenerationLLMClientAdapter(
        build_provider_judge_client(llm_client_config)
    )


def build_hijackrag_generator(
    config: Mapping[str, Any],
    *,
    target_system: HTTPTargetSystemAdapter,
    base_dir: Path | None = None,
) -> HijackRAGGenerator:
    attack_conf = attack_section(config, "hijackrag")
    llm_client = build_hijackrag_llm_client(config, attack_config=attack_conf)
    segments_path = _hijack_segments_path(attack_conf, base_dir=base_dir)
    correct_answer_provider = None
    if str(attack_conf.get("correct_answer_provider", "")).strip() == "target_system":
        correct_answer_provider = TargetCorrectAnswerProvider(
            target_system=target_system,
            attack="hijackrag",
        )
    if not segments_path.is_file():
        raise ScanConfigError(
            f"attack.hijackrag segments file not found: {segments_path}"
        )
    return HijackRAGGenerator.from_segments_file(
        segments_path,
        llm_client,
        correct_answer_provider=correct_answer_provider,
    )


def build_hijackrag_requests(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> tuple[HijackRAGRequest, ...]:
    attack_conf = attack_section(config, "hijackrag")
    case_configs = [
        *_inline_case_configs_hijack(attack_conf),
        *_case_file_configs_hijack(attack_conf, base_dir=base_dir),
    ]
    if not case_configs:
        raise ScanConfigError(
            "attack.hijackrag.cases or attack.hijackrag.case_files "
            "must contain at least one case"
        )
    return tuple(
        _build_hijackrag_request(
            attack_conf,
            case_config,
            case_number=case_number,
        )
        for case_number, case_config in enumerate(case_configs, start=1)
    )


def _hijack_segment_ids_from_case(
    case_config: Mapping[str, Any],
    prefix: str,
) -> tuple[str, ...]:
    raw = case_config.get("segment_ids", ())
    if raw in (None, "", ()):
        return ()
    if isinstance(raw, str):
        stripped = raw.strip()
        return (stripped,) if stripped else ()
    if isinstance(raw, bytes):
        raise ScanConfigError(
            f"{prefix}.segment_ids must be a string or a list of strings"
        )
    if not isinstance(raw, Sequence):
        raise ScanConfigError(
            f"{prefix}.segment_ids must be a string or a list of strings"
        )
    ids: list[str] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ScanConfigError(
                f"{prefix}.segment_ids[{index}] must be a non-empty string"
            )
        ids.append(item.strip())
    return tuple(ids)


def _build_hijackrag_request(
    attack_config: Mapping[str, Any],
    case_config: Mapping[str, Any],
    *,
    case_number: int,
) -> HijackRAGRequest:
    prefix = f"attack.hijackrag.cases[{case_number}]"
    insert_raw = case_config.get("hijack_insert", case_config.get("insert_prompt"))
    if not isinstance(insert_raw, str) or not insert_raw.strip():
        raise ScanConfigError(f"{prefix}.hijack_insert is required")
    case_config_accessor = ConfigAccessor(case_config, prefix=prefix)
    attack_config_accessor = ConfigAccessor(attack_config, prefix="attack.hijackrag")
    return HijackRAGRequest(
        query=case_config_accessor.get_required_string("query"),
        hijack_insert=insert_raw.strip(),
        correct_answer=case_config_accessor.get_required_string("correct_answer"),
        case_id=case_config_accessor.get_optional_string(
            "case_id", case_config_accessor.get_optional_string("id")
        ),
        adv_per_query=case_config_accessor.get_optional_int("adv_per_query", 1),
        segment_ids=_hijack_segment_ids_from_case(case_config, prefix),
        correct_answer_style=correct_answer_style_option(
            case_config,
            default=correct_answer_style_option(
                attack_config,
                prefix="attack.hijackrag",
            ),
            prefix=prefix,
        ),
        seed=case_config_accessor.get_optional_int(
            "seed", attack_config_accessor.get_optional_int("seed")
        ),
    )


def _inline_case_configs_hijack(
    attack_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_cases = attack_config.get("cases", ())
    return case_configs_from_value(raw_cases, "attack.hijackrag.cases")


def _case_file_configs_hijack(
    attack_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    file_paths = [
        *path_strings_from_value(
            attack_config.get("case_files", ()),
            "attack.hijackrag.case_files",
        ),
        *path_strings_from_value(
            attack_config.get("cases_file", ()),
            "attack.hijackrag.cases_file",
        ),
    ]
    cases: list[Mapping[str, Any]] = []
    for file_path in file_paths:
        cases.extend(load_case_configs(file_path, base_dir=base_dir))
    return tuple(cases)


def build_hijackrag_scan_config(config: Mapping[str, Any]) -> HijackRAGScanConfig:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise ScanConfigError("scan must be a mapping")
    scan_config_accessor = ConfigAccessor(scan_config, prefix="scan")
    return HijackRAGScanConfig(
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
    registry.register(HIJACK_PLUGIN)
    target_system = build_target_system(config)
    generator = build_hijackrag_generator(
        config,
        target_system=target_system,
        base_dir=base_dir,
    )
    evaluator = build_evaluator(
        config,
        attack_id="hijackrag",
        registry=registry,
    )
    return ConfiguredScanCommand(
        runner=HijackRAGScanRunner(
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=build_corpus_poisoner(config, base_dir=base_dir),
        ),
        requests=build_hijackrag_requests(config, base_dir=base_dir),
        scan_config=build_hijackrag_scan_config(config),
    )


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticHijackRAGCaseGenerator:
    attack_conf = attack_section(config, "hijackrag")
    llm_client = build_hijackrag_llm_client(config, attack_config=attack_conf)
    return AutomaticHijackRAGCaseGenerator(llm_client=llm_client)


def _serialize_case_for_yaml(case: Any) -> Mapping[str, Any]:
    case_id = str(getattr(case, "case_id", "") or "").strip()
    query = str(getattr(case, "query", "") or "").strip()
    correct_answer = str(getattr(case, "correct_answer", "") or "").strip()
    hijack_insert = str(getattr(case, "hijack_insert", "") or "").strip()
    adv_per_query = int(getattr(case, "adv_per_query", 1) or 1)
    if not query or not correct_answer or not hijack_insert:
        raise ScanConfigError("generated HijackRAG case is missing required fields")
    if not case_id:
        case_id = f"generated_case_{abs(hash(query)) % 1_000_000}"
    row: dict[str, Any] = {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "hijack_insert": hijack_insert,
        "adv_per_query": adv_per_query,
    }
    seed = getattr(case, "seed", None)
    if seed is not None:
        row["seed"] = int(seed)
    return row


def _generate_cases(
    config: Mapping[str, Any],
    params: GenerateCasesParams,
) -> list[Any]:
    generator = _build_automatic_case_generator(config)
    return list(
        generator.generate_cases(
            count=params.count,
            topic=params.topic,
            correct_answer_style=params.target_style,
            adv_per_query=params.adv_per_query,
            seed=params.seed,
        )
    )


HIJACK_PLUGIN = AttackPlugin(
    attack_id="hijackrag",
    display_name="HijackRAG",
    build_scan_command=_build_scan_command,
    judge_prompt_builder_factory=lambda: HijackRAGJudgePromptBuilder(),
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
)
