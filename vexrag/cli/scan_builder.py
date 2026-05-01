import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vexrag.attack_algorithms.poisonedrag import (
    PoisonedRAGGenerator,
    PoisonedRAGJudgePromptBuilder,
    PoisonedRAGRequest,
    PoisonedRAGScanConfig,
    PoisonedRAGScanRunner,
)
from vexrag.cli.errors import CLIConfigError, EvaluationConfigError
from vexrag.core.evaluation import (
    CosineSimilarityMetric,
    EvaluationStrategyProtocol,
    JudgePromptBuilderProtocol,
    LLMJudgeEvaluator,
    SemanticSimilarityEvaluator,
)
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
    TargetSystemQuery,
)


class ScanCommand(Protocol):
    """Runnable scan command built from CLI config."""

    def run(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class ConfiguredScanCommand:
    """Fully wired scan command built from CLI config."""

    runner: Any
    requests: Any
    scan_config: Any

    def run(self):
        return self.runner.run(self.requests, self.scan_config)


AttackScanBuilder = Callable[[Mapping[str, Any], Path | None], ScanCommand]
PromptBuilderFactory = Callable[[], JudgePromptBuilderProtocol]


def build_poisonedrag_scan_command(
    config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> ScanCommand:
    """Build a PoisonedRAG scan command from CLI config."""

    target_system = build_target_system(config)
    generator = build_poisonedrag_generator(config, target_system=target_system)
    evaluation_strategy = build_evaluation_strategy(config, attack="poisonedrag")
    return ConfiguredScanCommand(
        runner=PoisonedRAGScanRunner(
            generator=generator,
            target_system=target_system,
            evaluation_strategy=evaluation_strategy,
            corpus_poisoner=build_corpus_poisoner(config, base_dir=base_dir),
        ),
        requests=build_poisonedrag_requests(config, base_dir=base_dir),
        scan_config=build_poisonedrag_scan_config(config),
    )


ATTACK_SCAN_BUILDERS: Mapping[str, AttackScanBuilder] = {
    "poisonedrag": build_poisonedrag_scan_command,
}

JUDGE_PROMPT_BUILDERS: Mapping[str, PromptBuilderFactory] = {
    "poisonedrag": PoisonedRAGJudgePromptBuilder,
}


def build_scan_command(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> ScanCommand:
    """Build the configured scan command from the configured attack."""

    method = resolve_attack_method(config)
    try:
        builder = ATTACK_SCAN_BUILDERS[method]
    except KeyError as exc:
        raise CLIConfigError(
            "attack must configure a supported attack: "
            f"{_supported_names(ATTACK_SCAN_BUILDERS)}"
        ) from exc
    return builder(config, base_dir)


def build_target_system(config: Mapping[str, Any]) -> HTTPTargetSystemAdapter:
    """Build the configured target RAG system adapter."""

    target_config = _target_system_section(config)
    transport = str(target_config.get("transport", "http")).strip()
    if transport != "http":
        raise CLIConfigError("target_system.transport must be 'http'")

    http_config = target_config.get("http", target_config)
    if not isinstance(http_config, Mapping):
        raise CLIConfigError("target_system.http must be a mapping")

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


def build_poisonedrag_generator(
    config: Mapping[str, Any],
    *,
    target_system: HTTPTargetSystemAdapter,
) -> PoisonedRAGGenerator:
    """Build the PoisonedRAG generator with its configured LLM client."""

    attack_config = _attack_section(config, "poisonedrag")
    llm_client_config = _attack_llm_client_section(
        config,
        attack_config,
        attack="poisonedrag",
    )
    correct_answer_provider = None
    if str(attack_config.get("correct_answer_provider", "")).strip() == "target_system":
        correct_answer_provider = _TargetCorrectAnswerProvider(
            target_system=target_system,
            attack="poisonedrag",
        )

    return PoisonedRAGGenerator(
        llm_client=_JSONLLMClientAdapter(
            build_provider_judge_client(llm_client_config)
        ),
        correct_answer_provider=correct_answer_provider,
    )


def build_poisonedrag_requests(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> tuple[PoisonedRAGRequest, ...]:
    attack_config = _attack_section(config, "poisonedrag")
    case_configs = [
        *_inline_case_configs(attack_config),
        *_case_file_configs(attack_config, base_dir=base_dir),
    ]
    if not case_configs:
        raise CLIConfigError(
            "attack.poisonedrag.cases or attack.poisonedrag.case_files "
            "must contain at least one case"
        )

    return tuple(
        _build_poisonedrag_request(
            attack_config,
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
    return PoisonedRAGRequest(
        query=_required_string(case_config, "query", prefix),
        case_id=_optional_string(case_config.get("case_id", case_config.get("id"))),
        correct_answer=_optional_string(case_config.get("correct_answer")),
        target_incorrect_answer=_optional_string(
            case_config.get("target_incorrect_answer")
        ),
        adv_per_query=_int_option(
            case_config,
            "adv_per_query",
            _int_option(attack_config, "adv_per_query", 3),
        ),
        target_style=_target_style_option(
            case_config,
            default=_target_style_option(attack_config),
            prefix=prefix,
        ),
        poisoning_style=_poisoning_style_option(
            case_config,
            default=_poisoning_style_option(attack_config),
            prefix=prefix,
        ),
        seed=_optional_int(
            case_config.get("seed", attack_config.get("seed")),
            f"{prefix}.seed",
        ),
    )


def _inline_case_configs(
    attack_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_cases = attack_config.get("cases", ())
    return _case_configs_from_value(raw_cases, "attack.poisonedrag.cases")


def _case_file_configs(
    attack_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    file_paths = [
        *_path_strings_from_value(
            attack_config.get("case_files", ()),
            "attack.poisonedrag.case_files",
        ),
        *_path_strings_from_value(
            attack_config.get("cases_file", ()),
            "attack.poisonedrag.cases_file",
        ),
    ]

    cases: list[Mapping[str, Any]] = []
    for file_path in file_paths:
        cases.extend(_load_case_configs(file_path, base_dir=base_dir))
    return tuple(cases)


def _path_strings_from_value(value: Any, prefix: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise CLIConfigError(f"{prefix} must be a string or a list of strings")

    paths: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise CLIConfigError(f"{prefix}[{index}] must be a non-empty string")
        paths.append(item.strip())
    return tuple(paths)


def _load_case_configs(
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
        raise CLIConfigError(f"could not read cases file: {path}") from exc

    try:
        loaded = _load_case_file(raw_cases, path=path)
    except ValueError as exc:
        raise CLIConfigError(f"could not parse cases file: {path}") from exc

    if isinstance(loaded, Mapping) and "cases" in loaded:
        loaded = loaded["cases"]
    return _case_configs_from_value(loaded, str(path))


def _load_case_file(raw_cases: str, *, path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw_cases.splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(raw_cases)

    try:
        import yaml
    except ImportError as exc:
        raise CLIConfigError("PyYAML is required to read YAML cases files") from exc
    try:
        return yaml.safe_load(raw_cases)
    except yaml.YAMLError as exc:
        raise ValueError("cases file is invalid YAML") from exc


def _case_configs_from_value(value: Any, prefix: str) -> tuple[Mapping[str, Any], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CLIConfigError(f"{prefix} must be a list of case mappings")

    cases: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise CLIConfigError(f"{prefix}[{index}] must be a mapping")
        cases.append(item)
    return tuple(cases)


def build_poisonedrag_scan_config(
    config: Mapping[str, Any],
) -> PoisonedRAGScanConfig:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise CLIConfigError("scan must be a mapping")
    return PoisonedRAGScanConfig(
        repetitions=_int_option(scan_config, "repetitions", 1),
        attack_success_rate_threshold=_float_option(
            scan_config,
            "attack_success_rate_threshold",
            0.0,
        ),
        override_contexts=_bool_option(scan_config, "override_contexts", False),
        cleanup=_cleanup_option(scan_config),
    )


def build_corpus_poisoner(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> CorpusPoisoningAdapterProtocol | None:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise CLIConfigError("scan must be a mapping")

    poison_config = scan_config.get("corpus_poisoning", config.get("retrieval"))
    if poison_config in (None, False):
        return None
    if not isinstance(poison_config, Mapping):
        raise CLIConfigError("scan.corpus_poisoning must be a mapping")

    try:
        backend = RetrievalBackend(
            str(poison_config.get("backend", poison_config.get("type", "file_text")))
        )
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in RetrievalBackend)
        raise CLIConfigError(
            f"scan.corpus_poisoning.backend must be one of: {supported}"
        ) from exc

    if backend is not RetrievalBackend.FILE_TEXT:
        raise CLIConfigError(
            f"corpus poisoning for retrieval backend '{backend.value}' is not implemented"
        )

    file_text_config = _backend_config(poison_config, backend)
    corpus_path = _path_option(
        file_text_config,
        ("path", "directory", "contexts_dir", "corpus_path"),
        "scan.corpus_poisoning.file_text",
        base_dir=base_dir,
    )
    return FileTextCorpusPoisoningAdapter(path=corpus_path)


def build_evaluation_strategy(
    config: Mapping[str, Any],
    *,
    attack: str | None = None,
) -> EvaluationStrategyProtocol:
    """Build the configured CLI evaluation strategy."""

    resolved_attack = attack or resolve_attack_method(config)
    evaluation_config = _evaluation_section(config)
    strategy = str(evaluation_config.get("strategy", "semantic_similarity")).strip()
    if strategy == "semantic_similarity":
        return _build_semantic_similarity(evaluation_config)
    if strategy == "llm_judge":
        return _build_llm_judge(evaluation_config, attack=resolved_attack)
    raise EvaluationConfigError(
        "evaluation.strategy must be one of: semantic_similarity, llm_judge"
    )


def resolve_attack_method(config: Mapping[str, Any]) -> str:
    attack = config.get("attack")
    if not isinstance(attack, Mapping):
        raise CLIConfigError("attack must configure exactly one attack")

    methods = [name for name, value in attack.items() if isinstance(value, Mapping)]
    if len(methods) != 1:
        raise CLIConfigError("attack must configure exactly one attack")

    method = methods[0]
    if not isinstance(method, str) or not method.strip():
        raise CLIConfigError("attack names must be non-empty strings")

    return method.strip()


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
    attack: str,
) -> LLMJudgeEvaluator:
    strategy_config = _strategy_section(evaluation_config, "llm_judge")
    judge_config = _client_section(strategy_config, evaluation_config, "judge_client")
    return LLMJudgeEvaluator(
        judge_client=build_provider_judge_client(judge_config),
        prompt_builder=_build_prompt_builder(attack),
    )


def _build_prompt_builder(attack: str) -> JudgePromptBuilderProtocol:
    try:
        factory = JUDGE_PROMPT_BUILDERS[attack]
    except KeyError as exc:
        raise EvaluationConfigError(
            "llm_judge must configure a supported attack: "
            f"{_supported_names(JUDGE_PROMPT_BUILDERS)}"
        ) from exc
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
        raise EvaluationConfigError(f"{key} must be a number or null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigError(f"{key} must be a number or null") from exc


def _target_system_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    target_config = config.get("target_system")
    if not isinstance(target_config, Mapping):
        raise CLIConfigError("target_system must be configured")
    return target_config


def _attack_section(config: Mapping[str, Any], attack_name: str) -> Mapping[str, Any]:
    attack = config.get("attack")
    if not isinstance(attack, Mapping):
        raise CLIConfigError(f"attack.{attack_name} must be configured")
    attack_config = attack.get(attack_name)
    if not isinstance(attack_config, Mapping):
        raise CLIConfigError(f"attack.{attack_name} must be configured")
    return attack_config


def _attack_llm_client_section(
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
        raise CLIConfigError(f"attack.{attack}.llm_client must be configured")
    return client_config


def _required_string(config: Mapping[str, Any], key: str, prefix: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CLIConfigError(f"{prefix}.{key} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CLIConfigError("optional string values must be strings")
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
        raise CLIConfigError(f"{prefix}.{key} must be a mapping")
    return value


def _string_mapping_option(
    config: Mapping[str, Any],
    key: str,
    prefix: str,
) -> Mapping[str, str]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise CLIConfigError(f"{prefix}.{key} must be a mapping")
    return {str(name): str(item) for name, item in value.items()}


def _int_option(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise CLIConfigError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CLIConfigError(f"{key} must be an integer") from exc


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CLIConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CLIConfigError(f"{name} must be an integer") from exc


def _bool_option(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise CLIConfigError(f"{key} must be a boolean")
    return value


def _target_style_option(
    config: Mapping[str, Any],
    *,
    default: str = "short_fact",
    prefix: str = "attack.poisonedrag",
) -> str:
    value = str(config.get("target_style", default)).strip()
    if value not in {"short_fact", "paragraph"}:
        raise CLIConfigError(
            f"{prefix}.target_style must be 'short_fact' or 'paragraph'"
        )
    return value


def _poisoning_style_option(
    config: Mapping[str, Any],
    *,
    default: str = "original",
    prefix: str = "attack.poisonedrag",
) -> str:
    value = str(config.get("poisoning_style", default)).strip().lower()
    if value not in {"original", "aggressive", "soft"}:
        raise CLIConfigError(
            f"{prefix}.poisoning_style must be 'original', 'aggressive' or 'soft'"
        )
    return value


def _path_option(
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
    raise CLIConfigError(f"{prefix} must configure one of: {expected}")


def _backend_config(
    config: Mapping[str, Any],
    backend: RetrievalBackend,
) -> Mapping[str, Any]:
    nested = config.get(backend.value)
    if nested is None:
        return config
    if not isinstance(nested, Mapping):
        raise CLIConfigError(f"scan.corpus_poisoning.{backend.value} must be a mapping")
    return {**config, **nested}


def _cleanup_option(scan_config: Mapping[str, Any]) -> bool:
    poison_config = scan_config.get("corpus_poisoning", scan_config.get("retrieval"))
    if poison_config in (None, False):
        return False
    if not isinstance(poison_config, Mapping):
        raise CLIConfigError("scan.corpus_poisoning must be a mapping")
    return _bool_option(poison_config, "cleanup", False)


@dataclass(frozen=True, slots=True)
class _JSONLLMClientAdapter:
    client: Any

    @property
    def model_id(self) -> str:
        return str(getattr(self.client, "model", getattr(self.client, "model_id", "")))

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
        seed: int | None = None,
    ) -> str | Mapping[str, Any]:
        return self.client.complete_json(prompt)


@dataclass(frozen=True, slots=True)
class _TargetCorrectAnswerProvider:
    target_system: HTTPTargetSystemAdapter
    attack: str

    def get_correct_answer(
        self,
        query: str,
        *,
        target_style: str,
        seed: int | None = None,
    ) -> str:
        response = self.target_system.answer(
            TargetSystemQuery(
                query=query,
                metadata={
                    "attack": self.attack,
                    "purpose": "correct_answer",
                    "target_style": target_style,
                    "seed": seed,
                },
            )
        )
        return response.answer


def _supported_names(registry: Mapping[str, Any]) -> str:
    return ", ".join(sorted(registry))
