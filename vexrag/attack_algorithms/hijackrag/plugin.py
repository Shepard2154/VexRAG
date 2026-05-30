from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from vexrag.attack_algorithms.hijackrag.case_generator import (
    AutomaticHijackRAGCaseGenerator,
)
from vexrag.attack_algorithms.hijackrag.generator import HijackRAGGenerator
from vexrag.attack_algorithms.hijackrag.prompts import HijackRAGJudgePromptBuilder
from vexrag.attack_algorithms.hijackrag.scan_profile import (
    HIJACKRAG_ATTACK_ID,
    HIJACKRAG_SCAN_PROFILE,
)
from vexrag.attack_algorithms.hijackrag.schema import HijackRAGRequest
from vexrag.attack_algorithms.hijackrag.segments import default_hijack_segments_path
from vexrag.attack_algorithms.poison_base.case_id import stable_generated_case_id
from vexrag.attack_algorithms.poison_base.plugin_factory import (
    CorpusPoisonAttackSpec,
    build_attack_method_configurator,
)
from vexrag.attack_algorithms.poison_base.plugin_helpers import (
    build_attack_llm_client,
    load_attack_case_configs,
)
from vexrag.core.attack_configurator import TargetStyle
from vexrag.core.attack_configurator.types import GenerateCasesParams
from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.builder import (
    attack_section,
    correct_answer_style_option,
)
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError
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
            f"attack.{HIJACKRAG_ATTACK_ID}.segments_file must be a non-empty string when set"
        )
    path = Path(raw.strip())
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ScanConfigError(
            f"attack.{HIJACKRAG_ATTACK_ID}.segments_file not found: {path}"
        )
    return path


def build_hijackrag_generator(
    config: Mapping[str, Any],
    target_system: HTTPTargetSystemAdapter,
    base_dir: Path | None,
    registries: ScanRegistries | None,
) -> HijackRAGGenerator:
    attack_conf = attack_section(config, HIJACKRAG_ATTACK_ID)
    llm_client = build_attack_llm_client(
        config,
        attack_id=HIJACKRAG_ATTACK_ID,
        attack_config=attack_conf,
        registries=registries,
    )
    segments_path = _hijack_segments_path(attack_conf, base_dir=base_dir)
    correct_answer_provider = None
    if str(attack_conf.get("correct_answer_provider", "")).strip() == "target_system":
        correct_answer_provider = TargetCorrectAnswerProvider(
            target_system=target_system,
            attack=HIJACKRAG_ATTACK_ID,
        )
    return HijackRAGGenerator.from_segments_file(
        segments_path,
        llm_client,
        correct_answer_provider=correct_answer_provider,
    )


def build_hijackrag_requests(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> tuple[HijackRAGRequest, ...]:
    attack_conf = attack_section(config, HIJACKRAG_ATTACK_ID)
    case_configs = load_attack_case_configs(
        attack_conf,
        attack_id=HIJACKRAG_ATTACK_ID,
        base_dir=base_dir,
    )
    if not case_configs:
        raise ScanConfigError(
            f"attack.{HIJACKRAG_ATTACK_ID}.cases or "
            f"attack.{HIJACKRAG_ATTACK_ID}.case_files "
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
    prefix = f"attack.{HIJACKRAG_ATTACK_ID}.cases[{case_number}]"
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
        prefix=f"attack.{HIJACKRAG_ATTACK_ID}",
        error_type=ScanConfigError,
    )
    return HijackRAGRequest(
        query=case_config_accessor.get_required_string("query"),
        hijack_insert=insert_raw.strip(),
        correct_answer=case_config_accessor.get_required_string("correct_answer"),
        case_id=case_config_accessor.get_optional_string(
            "case_id", case_config_accessor.get_optional_string("id")
        ),
        adv_per_query=case_config_accessor.get_int(
            "adv_per_query", HIJACKRAG_SCAN_PROFILE.default_adv_per_query
        ),
        segment_ids=_hijack_segment_ids_from_case(case_config, prefix),
        correct_answer_style=correct_answer_style_option(
            case_config,
            default=correct_answer_style_option(
                attack_config,
                prefix=f"attack.{HIJACKRAG_ATTACK_ID}",
            ),
            prefix=prefix,
        ),
        seed=case_config_accessor.get_optional_int(
            "seed", attack_config_accessor.get_optional_int("seed")
        ),
    )


def _build_automatic_case_generator(
    config: Mapping[str, Any],
) -> AutomaticHijackRAGCaseGenerator:
    attack_conf = attack_section(config, HIJACKRAG_ATTACK_ID)
    llm_client = build_attack_llm_client(
        config,
        attack_id=HIJACKRAG_ATTACK_ID,
        attack_config=attack_conf,
    )
    return AutomaticHijackRAGCaseGenerator(llm_client=llm_client)


def _serialize_case_for_yaml(case: Any) -> Mapping[str, Any]:
    case_id = str(getattr(case, "case_id", "") or "").strip()
    query = str(getattr(case, "query", "") or "").strip()
    correct_answer = str(getattr(case, "correct_answer", "") or "").strip()
    hijack_insert = str(getattr(case, "hijack_insert", "") or "").strip()
    adv_per_query = int(
        getattr(case, "adv_per_query", HIJACKRAG_SCAN_PROFILE.default_adv_per_query)
        or HIJACKRAG_SCAN_PROFILE.default_adv_per_query
    )
    if not query or not correct_answer or not hijack_insert:
        raise ScanConfigError("generated HijackRAG case is missing required fields")
    if not case_id:
        case_id = stable_generated_case_id(query)
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
            correct_answer_style=cast(TargetStyle, params.target_style),
            adv_per_query=params.adv_per_query,
            seed=params.seed,
        )
    )


HIJACKRAG_SPEC = CorpusPoisonAttackSpec(
    attack_id=HIJACKRAG_ATTACK_ID,
    display_name="HijackRAG",
    scan_profile=HIJACKRAG_SCAN_PROFILE,
    build_generator=build_hijackrag_generator,
    build_requests=build_hijackrag_requests,
    build_automatic_case_generator=_build_automatic_case_generator,
    serialize_case_for_yaml=_serialize_case_for_yaml,
    generate_cases=_generate_cases,
    judge_prompt_builder_factory=lambda: HijackRAGJudgePromptBuilder(),
)

HIJACKRAG_PLUGIN = build_attack_method_configurator(HIJACKRAG_SPEC)
