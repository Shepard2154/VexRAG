from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.attack_algorithms.hijackrag.case_generator import (
    AutomaticHijackRAGCaseGenerator,
)
from vexrag.attack_algorithms.hijackrag.evaluation import HijackRAGJudgePromptBuilder
from vexrag.attack_algorithms.hijackrag.generator import HijackRAGGenerator
from vexrag.attack_algorithms.hijackrag.scan import HijackRAGScanRunner
from vexrag.attack_algorithms.hijackrag.schema import HijackRAGRequest
from vexrag.attack_algorithms.hijackrag.segments import default_hijack_segments_path
from vexrag.attack_algorithms.poison_base.plugin_helpers import (
    build_attack_llm_client,
    load_attack_case_configs,
)
from vexrag.attack_algorithms.poison_base.scan_config import (
    build_corpus_poison_scan_config,
)
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
    correct_answer_style_option,
)
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.scan.execution import ConfiguredScanCommand
from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    TargetCorrectAnswerProvider,
)


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


def build_hijackrag_generator(
    config: Mapping[str, Any],
    *,
    target_system: HTTPTargetSystemAdapter,
    base_dir: Path | None = None,
    registries: ScanRegistries | None = None,
) -> HijackRAGGenerator:
    attack_conf = attack_section(config, "hijackrag")
    llm_client = build_attack_llm_client(
        config,
        attack_id="hijackrag",
        attack_config=attack_conf,
        registries=registries,
    )
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
    case_configs = load_attack_case_configs(
        attack_conf,
        attack_id="hijackrag",
        base_dir=base_dir,
    )
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
    case_config_accessor = ConfigAccessor(
        case_config,
        prefix=prefix,
        error_type=ScanConfigError,
    )
    attack_config_accessor = ConfigAccessor(
        attack_config,
        prefix="attack.hijackrag",
        error_type=ScanConfigError,
    )
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


def _build_scan_command(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> ConfiguredScanCommand:
    registries = create_scan_registries(base_dir=base_dir)
    target_system = build_target_system(
        config,
        registry=registries.target_systems,
    )
    generator = build_hijackrag_generator(
        config,
        target_system=target_system,
        base_dir=base_dir,
        registries=registries,
    )
    evaluator = build_evaluator(
        config,
        attack_id="hijackrag",
        registries=registries,
    )
    return ConfiguredScanCommand(
        runner=HijackRAGScanRunner(
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=build_corpus_poisoner(config, registries=registries),
        ),
        requests=build_hijackrag_requests(config, base_dir=base_dir),
        scan_config=build_corpus_poison_scan_config(config),
    )


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticHijackRAGCaseGenerator:
    attack_conf = attack_section(config, "hijackrag")
    llm_client = build_attack_llm_client(
        config,
        attack_id="hijackrag",
        attack_config=attack_conf,
    )
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


HIJACK_PLUGIN = AttackMethodConfigurator(
    attack_id="hijackrag",
    display_name="HijackRAG",
    build_scan_command=_build_scan_command,
    judge_prompt_builder_factory=lambda: HijackRAGJudgePromptBuilder(),
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
)
