import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config_errors import EvaluationConfigError, ScanConfigError
from vexrag.core.evaluation import (
    CosineSimilarityMetric,
    EvaluationStrategyProtocol,
    JudgePromptBuilderProtocol,
    LLMJudgeEvaluator,
    SemanticSimilarityEvaluator,
)
from vexrag.core.evaluation.multi import MultiEvaluator
from vexrag.core.providers import (
    build_embedding_client as build_provider_embedding_client,
)
from vexrag.core.providers import (
    build_judge_client as build_provider_judge_client,
)
from vexrag.core.retrieval import (
    CorpusPoisoningAdapterProtocol,
    FileTextCorpusPoisoningAdapter,
    RetrievalBackend,
)
from vexrag.core.target import (
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
)


def attack_section(config: Mapping[str, Any], attack_name: str) -> Mapping[str, Any]:
    attack = config.get("attack")
    if not isinstance(attack, Mapping):
        raise ScanConfigError(f"attack.{attack_name} must be configured")
    attack_config = attack.get(attack_name)
    if not isinstance(attack_config, Mapping):
        raise ScanConfigError(f"attack.{attack_name} must be configured")
    return attack_config


def build_target_system(config: Mapping[str, Any]) -> HTTPTargetSystemAdapter:
    target_config = _target_system_section(config)
    transport = str(target_config.get("transport", "http")).strip()
    if transport != "http":
        raise ScanConfigError("target_system.transport must be 'http'")

    http_config = target_config.get("http", target_config)
    if not isinstance(http_config, Mapping):
        raise ScanConfigError("target_system.http must be a mapping")

    return HTTPTargetSystemAdapter(
        HTTPTargetSystemAdapterConfig(
            base_url=_required_string(http_config, "base_url", "target_system"),
            route=str(http_config.get("route", "")).strip(),
            method=str(http_config.get("method", "POST")).strip(),
            timeout=_optional_float_option(http_config, "timeout", 10.0),
            request_template=_mapping_option(
                http_config,
                "request_template",
                "target_system",
                default={"query": "{query}", "contexts": "{contexts}"},
            ),
            response_paths=_mapping_option(
                http_config,
                "response_paths",
                "target_system",
                default={"answer": "answer", "contexts": "contexts"},
            ),
            headers=_string_mapping_option(http_config, "headers", "target_system"),
        )
    )


def build_corpus_poisoner(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> CorpusPoisoningAdapterProtocol | None:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise ScanConfigError("scan must be a mapping")

    poison_config = scan_config.get("corpus_poisoning", config.get("retrieval"))
    if poison_config in (None, False):
        return None
    if not isinstance(poison_config, Mapping):
        raise ScanConfigError("scan.corpus_poisoning must be a mapping")

    try:
        backend = RetrievalBackend(
            str(poison_config.get("backend", poison_config.get("type", "file_text")))
        )
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in RetrievalBackend)
        raise ScanConfigError(
            f"scan.corpus_poisoning.backend must be one of: {supported}"
        ) from exc

    if backend is not RetrievalBackend.FILE_TEXT:
        raise ScanConfigError(
            f"corpus poisoning for retrieval backend '{backend.value}' is not implemented"
        )

    file_text_config = _backend_config(poison_config, backend)
    corpus_path = _path_option(
        file_text_config,
        ("path", "directory", "contexts_dir", "corpus_path"),
        "scan.corpus_poisoning.file_text",
        base_dir=base_dir,
    )
    prefix_raw = poison_config.get("filename_prefix", "poisonedrag")
    if not isinstance(prefix_raw, str) or not prefix_raw.strip():
        filename_prefix = "poisonedrag"
    else:
        filename_prefix = prefix_raw.strip()
    return FileTextCorpusPoisoningAdapter(
        path=corpus_path,
        filename_prefix=filename_prefix,
    )


def _build_one_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> EvaluationStrategyProtocol:
    strategy = str(evaluation_config.get("strategy", "semantic_similarity")).strip()
    if strategy == "semantic_similarity":
        return _build_semantic_similarity(evaluation_config)
    if strategy == "llm_judge":
        return _build_llm_judge(
            evaluation_config, attack_id=attack_id, registry=registry
        )
    raise EvaluationConfigError(
        "evaluation.strategy must be one of: semantic_similarity, llm_judge"
    )


def build_evaluation_strategy(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> EvaluationStrategyProtocol:
    has_bundle = "evaluations" in config
    has_single = "evaluation" in config
    if has_bundle and has_single:
        raise EvaluationConfigError(
            "use either top-level evaluation or evaluations, not both"
        )
    if has_bundle:
        bundle = config["evaluations"]
        if not isinstance(bundle, Mapping):
            raise EvaluationConfigError("evaluations must be a mapping")
        combine_raw = str(bundle.get("combine", "any")).strip().lower()
        if combine_raw not in ("any", "all"):
            raise EvaluationConfigError("evaluations.combine must be 'any' or 'all'")
        raw_list = bundle.get("evaluators")
        if not isinstance(raw_list, list) or not raw_list:
            raise EvaluationConfigError(
                "evaluations.evaluators must be a non-empty list of evaluator configs"
            )
        built: list[EvaluationStrategyProtocol] = []
        for index, item in enumerate(raw_list):
            if not isinstance(item, Mapping):
                raise EvaluationConfigError(
                    f"evaluations.evaluators[{index}] must be a mapping"
                )
            built.append(
                _build_one_evaluator(
                    item,
                    attack_id=attack_id,
                    registry=registry,
                )
            )
        return MultiEvaluator(tuple(built), combine=combine_raw)

    if not has_single:
        raise EvaluationConfigError("configure evaluation or evaluations")
    evaluation_config = _evaluation_section(config)
    return _build_one_evaluator(
        evaluation_config,
        attack_id=attack_id,
        registry=registry,
    )


def attack_llm_client_section(
    config: Mapping[str, Any],
    attack_config: Mapping[str, Any],
    *,
    attack: str,
) -> Mapping[str, Any]:
    client_config = (
        attack_config.get("llm_client")
        or attack_config.get("generator_client")
        or config.get("llm_client")
    )
    if not isinstance(client_config, Mapping):
        raise ScanConfigError(f"attack.{attack}.llm_client must be configured")
    return client_config


def _build_semantic_similarity(
    evaluation_config: Mapping[str, Any],
) -> SemanticSimilarityEvaluator:
    strategy_config = _strategy_section(evaluation_config, "semantic_similarity")
    metric_name = str(strategy_config.get("metric", "cosine")).strip()
    if metric_name != "cosine":
        raise EvaluationConfigError("semantic_similarity.metric must be 'cosine'")

    embedding_config = _client_section(
        strategy_config,
        evaluation_config,
        "embedding_client",
    )
    return SemanticSimilarityEvaluator(
        embedding_client=build_provider_embedding_client(embedding_config),
        metric=CosineSimilarityMetric(),
        attack_similarity_threshold=_float_option(
            strategy_config,
            "attack_similarity_threshold",
            0.75,
        ),
        max_reference_similarity=_float_option(
            strategy_config,
            "max_reference_similarity",
            0.6,
        ),
        attack_margin_threshold=_float_option(
            strategy_config,
            "attack_margin_threshold",
            0.1,
        ),
    )


def _build_llm_judge(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> LLMJudgeEvaluator:
    strategy_config = _strategy_section(evaluation_config, "llm_judge")
    judge_config = _client_section(strategy_config, evaluation_config, "judge_client")
    prompt_builder = _resolve_judge_prompt_builder(registry, attack_id)
    return LLMJudgeEvaluator(
        judge_client=build_provider_judge_client(judge_config),
        prompt_builder=prompt_builder,
    )


def _resolve_judge_prompt_builder(
    registry: AttackRegistry,
    attack_id: str,
) -> JudgePromptBuilderProtocol:
    factory = registry.get(attack_id).judge_prompt_builder_factory
    if factory is None:
        supported = ", ".join(
            aid
            for aid in registry.ids()
            if registry.get(aid).judge_prompt_builder_factory
        )
        raise EvaluationConfigError(
            "llm_judge is not supported for this attack or judge prompts are missing; "
            f"attacks with judge support: {supported}"
        )
    return factory()


def _evaluation_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "evaluation" not in config:
        return config
    evaluation = config["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise EvaluationConfigError("evaluation must be a mapping")
    return evaluation


def _strategy_section(
    evaluation_config: Mapping[str, Any],
    strategy: str,
) -> Mapping[str, Any]:
    nested = evaluation_config.get(strategy, {})
    if not isinstance(nested, Mapping):
        raise EvaluationConfigError(f"evaluation.{strategy} must be a mapping")
    return nested


def _client_section(
    strategy_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    config = strategy_config.get(key, evaluation_config.get(key))
    if not isinstance(config, Mapping):
        raise EvaluationConfigError(f"evaluation.{key} must be configured")
    return config


def _target_system_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    target_config = config.get("target_system")
    if not isinstance(target_config, Mapping):
        raise ScanConfigError("target_system must be configured")
    return target_config


def _required_string(config: Mapping[str, Any], key: str, prefix: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScanConfigError(f"{prefix}.{key} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ScanConfigError("optional string values must be strings")
    stripped = value.strip()
    return stripped or None


def _mapping_option(
    config: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = config.get(key, default)
    if not isinstance(value, Mapping):
        raise ScanConfigError(f"{prefix}.{key} must be a mapping")
    return value


def _string_mapping_option(
    config: Mapping[str, Any],
    key: str,
    prefix: str,
) -> Mapping[str, str]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ScanConfigError(f"{prefix}.{key} must be a mapping")
    return {str(name): str(item) for name, item in value.items()}


def _int_option(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ScanConfigError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{key} must be an integer") from exc


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{name} must be an integer") from exc


def _bool_option(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ScanConfigError(f"{key} must be a boolean")
    return value


def _float_option(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise EvaluationConfigError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigError(f"{key} must be a number") from exc


def _optional_float_option(
    config: Mapping[str, Any],
    key: str,
    default: float,
) -> float | None:
    value = config.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{key} must be a number or null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{key} must be a number or null") from exc


def target_style_option(
    config: Mapping[str, Any],
    *,
    default: str = "short_fact",
    prefix: str = "attack",
) -> str:
    value = str(config.get("target_style", default)).strip()
    if value not in {"short_fact", "paragraph"}:
        raise ScanConfigError(
            f"{prefix}.target_style must be 'short_fact' or 'paragraph'"
        )
    return value


def correct_answer_style_option(
    config: Mapping[str, Any],
    *,
    default: str = "short_fact",
    prefix: str = "attack",
) -> str:
    """Prefer YAML ``correct_answer_style``; fall back to legacy ``target_style``."""
    raw = config.get("correct_answer_style")
    if raw is None:
        raw = config.get("target_style", default)
    value = str(raw).strip()
    if value not in {"short_fact", "paragraph"}:
        raise ScanConfigError(
            f"{prefix}.correct_answer_style (or legacy target_style) must be "
            "'short_fact' or 'paragraph'"
        )
    return value


def poisoning_style_option(
    config: Mapping[str, Any],
    *,
    default: str = "original",
    prefix: str = "attack.poisonedrag",
) -> str:
    value = str(config.get("poisoning_style", default)).strip().lower()
    if value not in {"original", "aggressive", "soft"}:
        raise ScanConfigError(
            f"{prefix}.poisoning_style must be 'original', 'aggressive' or 'soft'"
        )
    return value


def path_option(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
    prefix: str,
    *,
    base_dir: Path | None,
) -> Path:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value.strip())
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            return path
    expected = ", ".join(keys)
    raise ScanConfigError(f"{prefix} must configure one of: {expected}")


def _path_option(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
    prefix: str,
    *,
    base_dir: Path | None,
) -> Path:
    return path_option(config, keys, prefix, base_dir=base_dir)


def _backend_config(
    config: Mapping[str, Any],
    backend: RetrievalBackend,
) -> Mapping[str, Any]:
    nested = config.get(backend.value)
    if nested is None:
        return config
    if not isinstance(nested, Mapping):
        raise ScanConfigError(
            f"scan.corpus_poisoning.{backend.value} must be a mapping"
        )
    return {**config, **nested}


def cleanup_option(scan_config: Mapping[str, Any]) -> bool:
    poison_config = scan_config.get("corpus_poisoning", scan_config.get("retrieval"))
    if poison_config in (None, False):
        return False
    if not isinstance(poison_config, Mapping):
        raise ScanConfigError("scan.corpus_poisoning must be a mapping")
    return _bool_option(poison_config, "cleanup", False)


def path_strings_from_value(value: Any, prefix: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise ScanConfigError(f"{prefix} must be a string or a list of strings")

    paths: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ScanConfigError(f"{prefix}[{index}] must be a non-empty string")
        paths.append(item.strip())
    return tuple(paths)


def case_configs_from_value(value: Any, prefix: str) -> tuple[Mapping[str, Any], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScanConfigError(f"{prefix} must be a list of case mappings")

    cases: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ScanConfigError(f"{prefix}[{index}] must be a mapping")
        cases.append(item)
    return tuple(cases)


def load_case_file(raw_cases: str, *, path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw_cases.splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(raw_cases)

    try:
        import yaml
    except ImportError as exc:
        raise ScanConfigError("PyYAML is required to read YAML cases files") from exc
    try:
        return yaml.safe_load(raw_cases)
    except yaml.YAMLError as exc:
        raise ValueError("cases file is invalid YAML") from exc


def load_case_configs(
    file_path: str,
    *,
    base_dir: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    path = Path(file_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    try:
        raw_cases = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScanConfigError(f"could not read cases file: {path}") from exc

    try:
        loaded = load_case_file(raw_cases, path=path)
    except ValueError as exc:
        raise ScanConfigError(f"could not parse cases file: {path}") from exc

    if isinstance(loaded, Mapping) and "cases" in loaded:
        loaded = loaded["cases"]
    return case_configs_from_value(loaded, str(path))
