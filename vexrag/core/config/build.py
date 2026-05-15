import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.evaluation.attack_verdict import CombineMode, EvaluationStrategy
from vexrag.core.evaluation.composite_evaluator import CompositeEvaluator
from vexrag.core.evaluation.cosine_similarity_metric import cosine_similarity
from vexrag.core.evaluation.embedding_similarity_evaluator import (
    EmbeddingSimilarityEvaluator,
)
from vexrag.core.evaluation.evaluator_protocols import (
    EmbeddingClient,
    Evaluator,
    JudgePromptBuilder,
)
from vexrag.core.evaluation.llm_judge_evaluator import LLMJudgeEvaluator
from vexrag.core.evaluation.provider_client_adapters import (
    ProviderBackedEmbeddingClient,
    ProviderBackedJudgeClient,
)
from vexrag.core.providers import (
    build_embedding_client as build_provider_embedding_client,
)
from vexrag.core.providers import (
    build_judge_client as build_provider_judge_client,
)
from vexrag.core.retrieval import (
    ChromaPoisoner,
    CorpusPoisoningAdapterProtocol,
    FaissPoisoner,
    FileTextPoisoner,
    QdrantPoisoner,
    RetrievalBackend,
)
from vexrag.core.target import (
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
)

from .errors import EvaluationConfigError, ScanConfigError
from .options import ConfigAccessor


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
    http_config_accessor = ConfigAccessor(http_config, prefix="target_system.http")

    scan_raw = config.get("scan", {})
    include_raw_payload = False
    if isinstance(scan_raw, Mapping):
        scan_accessor = ConfigAccessor(scan_raw, prefix="scan")
        include_raw_payload = scan_accessor.get_bool(
            "debug_include_raw_target_response",
            False,
        )

    return HTTPTargetSystemAdapter(
        HTTPTargetSystemAdapterConfig(
            base_url=http_config_accessor.get_required_string("base_url"),
            route=str(http_config.get("route", "")).strip(),
            method=str(http_config.get("method", "POST")).strip(),
            timeout=http_config_accessor.get_optional_float("timeout", 10.0),
            request_template=http_config_accessor.get_mapping(
                "request_template",
                {"query": "{query}", "contexts": "{contexts}"},
            ),
            response_paths=http_config_accessor.get_mapping(
                "response_paths",
                {"answer": "answer", "contexts": "contexts"},
            ),
            headers=http_config_accessor.get_string_mapping("headers"),
            include_raw_response_in_metadata=include_raw_payload,
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

    poison_config_accessor = ConfigAccessor(
        poison_config, prefix="scan.corpus_poisoning", base_dir=base_dir
    )

    try:
        backend = RetrievalBackend(
            str(poison_config.get("backend", poison_config.get("type", "file_text")))
        )
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in RetrievalBackend)
        raise ScanConfigError(
            f"scan.corpus_poisoning.backend must be one of: {supported}"
        ) from exc

    backend_config = _backend_config(poison_config, backend)
    prefix_raw = poison_config.get("filename_prefix", "poisonedrag")
    if not isinstance(prefix_raw, str) or not prefix_raw.strip():
        filename_prefix = "poisonedrag"
    else:
        filename_prefix = prefix_raw.strip()

    if backend is RetrievalBackend.FILE_TEXT:
        corpus_path = poison_config_accessor.get_path(
            "path", "directory", "contexts_dir", "corpus_path"
        )
        return FileTextPoisoner(
            path=corpus_path,
            filename_prefix=filename_prefix,
        )

    embedding_section = _corpus_poisoning_embedding_section(
        backend_config,
        f"scan.corpus_poisoning.{backend.value}",
    )
    embedding_client = build_provider_embedding_client(embedding_section)
    l2_normalize = poison_config_accessor.get_bool("l2_normalize", False)

    if backend is RetrievalBackend.QDRANT:
        return _build_qdrant_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    if backend is RetrievalBackend.CHROMA:
        return _build_chroma_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    if backend is RetrievalBackend.FAISS:
        return _build_faiss_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    raise ScanConfigError(f"unsupported corpus poisoning backend: {backend.value}")


def _corpus_poisoning_embedding_section(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> Mapping[str, Any]:
    emb = backend_cfg.get("embedding_client")
    if not isinstance(emb, Mapping):
        raise ScanConfigError(
            f"{prefix}.embedding_client must be configured "
            "(must match the target RAG embedding model and preprocessing)"
        )
    return emb


def _vector_backend_collection_name(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> str:
    for key in ("collection", "collection_name"):
        raw = backend_cfg.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise ScanConfigError(f"{prefix} must set collection or collection_name")


def _optional_poison_timeout(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> float | None:
    value = backend_cfg.get("timeout")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{prefix}.timeout must be a number or null")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{prefix}.timeout must be a number or null") from exc
    if out <= 0:
        raise ScanConfigError(f"{prefix}.timeout must be greater than 0")
    return out


def _build_qdrant_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> QdrantPoisoner:
    prefix = "scan.corpus_poisoning.qdrant"
    qdrant_config_accessor = ConfigAccessor(
        backend_cfg, prefix="scan.corpus_poisoning.qdrant"
    )
    url = qdrant_config_accessor.get_optional_string("url")
    path_str = backend_cfg.get("path")
    path: Path | None = None
    if isinstance(path_str, str) and path_str.strip():
        path = Path(path_str.strip())
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
    if url and path:
        raise ScanConfigError(f"{prefix} must set only one of url or path")
    if not url and not path:
        raise ScanConfigError(f"{prefix} must set url or path")
    collection = _vector_backend_collection_name(backend_cfg, prefix)
    vector_name = qdrant_config_accessor.get_optional_string("vector_name")
    timeout = _optional_poison_timeout(backend_cfg, prefix)
    api_key = qdrant_config_accessor.get_optional_string("api_key")
    return QdrantPoisoner(
        url=url,
        path=path,
        collection=collection,
        embedding_client=embedding_client,
        vector_name=vector_name,
        timeout=timeout,
        api_key=api_key,
        l2_normalize=l2_normalize,
    )


def _build_chroma_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> ChromaPoisoner:
    prefix = "scan.corpus_poisoning.chroma"
    chroma_config_accessor = ConfigAccessor(
        backend_cfg, prefix="scan.corpus_poisoning.chroma"
    )
    host = chroma_config_accessor.get_optional_string("host")
    port = chroma_config_accessor.get_optional_int("port", 8000)
    path_str = backend_cfg.get("persist_directory") or backend_cfg.get("path")
    persist_directory: Path | None = None
    if host:
        if isinstance(path_str, str) and path_str.strip():
            raise ScanConfigError(
                f"{prefix} cannot set persist_directory/path when host is set"
            )
    else:
        if not isinstance(path_str, str) or not path_str.strip():
            raise ScanConfigError(
                f"{prefix} must set persist_directory or path for local Chroma"
            )
        persist_directory = Path(path_str.strip())
        if not persist_directory.is_absolute() and base_dir is not None:
            persist_directory = base_dir / persist_directory
    collection_name = _vector_backend_collection_name(backend_cfg, prefix)
    return ChromaPoisoner(
        persist_directory=persist_directory,
        host=host,
        port=port,
        collection_name=collection_name,
        embedding_client=embedding_client,
        l2_normalize=l2_normalize,
    )


def _build_faiss_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> FaissPoisoner:
    prefix = "scan.corpus_poisoning.faiss"
    faiss_config_accessor = ConfigAccessor(
        backend_cfg, prefix="scan.corpus_poisoning.faiss"
    )
    faiss_dir = faiss_config_accessor.get_path(
        backend_cfg,
        ("directory", "path", "faiss_dir"),
        prefix,
        base_dir=base_dir,
    )
    poison_start = backend_cfg.get("poison_id_start")
    poison_id_start = -1
    if poison_start is not None:
        if isinstance(poison_start, bool):
            raise ScanConfigError(f"{prefix}.poison_id_start must be an integer")
        try:
            poison_id_start = int(poison_start)
        except (TypeError, ValueError) as exc:
            raise ScanConfigError(
                f"{prefix}.poison_id_start must be an integer"
            ) from exc
    return FaissPoisoner(
        faiss_dir,
        embedding_client,
        l2_normalize=l2_normalize,
        poison_id_start=poison_id_start,
    )


_COMPOSITE_STRATEGY = "composite"

_SIMILARITY_METRICS: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "cosine": cosine_similarity,
}

_EVALUATOR_BUILDERS: dict[
    EvaluationStrategy,
    Callable[..., Evaluator],
] = {}


def _parse_evaluation_strategy(evaluation_config: Mapping[str, Any]) -> str:
    return str(
        evaluation_config.get("strategy", EvaluationStrategy.EMBEDDING_SIMILARITY)
    ).strip()


def build_evaluator(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> Evaluator:
    if "evaluations" in config:
        raise EvaluationConfigError(
            "top-level 'evaluations' was removed; use evaluation.strategy: "
            "composite with evaluation.evaluators"
        )
    evaluation_config = _evaluation_section(config)
    strategy = _parse_evaluation_strategy(evaluation_config)
    if strategy == _COMPOSITE_STRATEGY:
        return _build_composite_evaluator(
            evaluation_config,
            attack_id=attack_id,
            registry=registry,
        )
    try:
        resolved = EvaluationStrategy(strategy)
    except ValueError as exc:
        supported = ", ".join(
            (
                _COMPOSITE_STRATEGY,
                EvaluationStrategy.EMBEDDING_SIMILARITY,
                EvaluationStrategy.LLM_JUDGE,
            )
        )
        raise EvaluationConfigError(
            f"evaluation.strategy must be one of: {supported}"
        ) from exc
    builder = _EVALUATOR_BUILDERS[resolved]
    return builder(
        evaluation_config,
        attack_id=attack_id,
        registry=registry,
    )


def _build_composite_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> CompositeEvaluator:
    combine_raw = str(evaluation_config.get("combine", CombineMode.ANY)).strip().lower()
    try:
        combine = CombineMode(combine_raw)
    except ValueError as exc:
        raise EvaluationConfigError(
            "evaluation.combine must be 'any' or 'all'"
        ) from exc
    raw_list = evaluation_config.get("evaluators")
    if not isinstance(raw_list, list) or not raw_list:
        raise EvaluationConfigError(
            "evaluation.evaluators must be a non-empty list when strategy is composite"
        )
    built: list[Evaluator] = []
    for index, item in enumerate(raw_list):
        if not isinstance(item, Mapping):
            raise EvaluationConfigError(
                f"evaluation.evaluators[{index}] must be a mapping"
            )
        built.append(
            _build_evaluator_from_section(
                item,
                attack_id=attack_id,
                registry=registry,
            )
        )
    return CompositeEvaluator(tuple(built), combine=combine)


def _build_evaluator_from_section(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> Evaluator:
    strategy = _parse_evaluation_strategy(evaluation_config)
    if strategy == _COMPOSITE_STRATEGY:
        raise EvaluationConfigError("nested composite evaluators are not supported")
    try:
        resolved = EvaluationStrategy(strategy)
    except ValueError as exc:
        supported = ", ".join(
            (
                EvaluationStrategy.EMBEDDING_SIMILARITY,
                EvaluationStrategy.LLM_JUDGE,
            )
        )
        raise EvaluationConfigError(
            f"evaluation.strategy must be one of: {supported}"
        ) from exc
    builder = _EVALUATOR_BUILDERS[resolved]
    return builder(
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


def _build_embedding_similarity_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> EmbeddingSimilarityEvaluator:
    del attack_id, registry
    strategy_config = _strategy_section(
        evaluation_config, EvaluationStrategy.EMBEDDING_SIMILARITY
    )
    metric_name = str(strategy_config.get("metric", "cosine")).strip()
    metric_fn = _SIMILARITY_METRICS.get(metric_name)
    if metric_fn is None:
        supported = ", ".join(sorted(_SIMILARITY_METRICS))
        raise EvaluationConfigError(
            f"evaluation.{EvaluationStrategy.EMBEDDING_SIMILARITY}.metric "
            f"must be one of: {supported}"
        )

    embedding_config = _client_section(
        strategy_config,
        evaluation_config,
        "embedding_client",
    )
    strategy_config_accessor = ConfigAccessor(
        strategy_config, prefix=f"evaluation.{EvaluationStrategy.EMBEDDING_SIMILARITY}"
    )
    return EmbeddingSimilarityEvaluator(
        embedding_client=ProviderBackedEmbeddingClient(
            build_provider_embedding_client(embedding_config)
        ),
        metric=metric_fn,
        attack_similarity_threshold=strategy_config_accessor.get_optional_float(
            "attack_similarity_threshold", 0.75
        ),
        max_reference_similarity=strategy_config_accessor.get_optional_float(
            "max_reference_similarity", 0.6
        ),
        attack_margin_threshold=strategy_config_accessor.get_optional_float(
            "attack_margin_threshold", 0.1
        ),
    )


def _build_llm_judge_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registry: AttackRegistry,
) -> LLMJudgeEvaluator:
    strategy_config = _strategy_section(evaluation_config, EvaluationStrategy.LLM_JUDGE)
    judge_config = _client_section(strategy_config, evaluation_config, "judge_client")
    prompt_builder = _resolve_judge_prompt_builder(registry, attack_id)
    return LLMJudgeEvaluator(
        judge_client=ProviderBackedJudgeClient(
            build_provider_judge_client(judge_config)
        ),
        prompt_builder=prompt_builder,
    )


_EVALUATOR_BUILDERS[EvaluationStrategy.EMBEDDING_SIMILARITY] = (
    _build_embedding_similarity_evaluator
)
_EVALUATOR_BUILDERS[EvaluationStrategy.LLM_JUDGE] = _build_llm_judge_evaluator


def _resolve_judge_prompt_builder(
    registry: AttackRegistry,
    attack_id: str,
) -> JudgePromptBuilder:
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
    nested = evaluation_config.get(strategy)
    if strategy == EvaluationStrategy.EMBEDDING_SIMILARITY and not nested:
        nested = evaluation_config.get(EvaluationStrategy.EMBEDDING_SIMILARITY)
    if nested is None:
        nested = {}
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
    poison_config_accessor = ConfigAccessor(
        poison_config, prefix="scan.corpus_poisoning"
    )
    return poison_config_accessor.get_bool("cleanup", False)


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
