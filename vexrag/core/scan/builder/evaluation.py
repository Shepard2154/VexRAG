from collections.abc import Mapping
from typing import Any

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.evaluation import (
    CombineMode,
    CompositeEvaluator,
    EmbeddingSimilarityEvaluator,
    EvaluationRegistry,
    EvaluationStrategy,
    Evaluator,
    LLMJudgeEvaluator,
    ProviderBackedEmbeddingClient,
    ProviderBackedJsonCompletionClient,
)
from vexrag.core.evaluation.contracts import JudgePromptBuilder
from vexrag.core.evaluation.metrics import cosine_similarity
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import EvaluationConfigError


def create_default_evaluation_registry() -> EvaluationRegistry:
    return EvaluationRegistry(
        evaluator_builders={
            EvaluationStrategy.EMBEDDING_SIMILARITY: build_embedding_similarity_evaluator,
            EvaluationStrategy.LLM_JUDGE: build_llm_judge_evaluator,
        },
        metrics={"cosine": cosine_similarity},
    )


def build_evaluator(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
) -> Evaluator:
    if "evaluations" in config:
        raise EvaluationConfigError(
            "top-level 'evaluations' was removed; use evaluation.strategy: "
            "composite with evaluation.evaluators"
        )
    evaluation_config = evaluation_section(config)
    strategy = parse_evaluation_strategy(evaluation_config)
    if strategy == EvaluationStrategy.COMPOSITE:
        return _build_composite_evaluator(
            evaluation_config,
            attack_id=attack_id,
            registries=registries,
        )
    builder = registries.evaluations.get_evaluator_builder(strategy)
    return builder(evaluation_config, attack_id=attack_id, registries=registries)


def parse_evaluation_strategy(evaluation_config: Mapping[str, Any]) -> str:
    return str(
        evaluation_config.get("strategy", EvaluationStrategy.EMBEDDING_SIMILARITY)
    ).strip()


def build_embedding_similarity_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
) -> EmbeddingSimilarityEvaluator:
    del attack_id
    strategy_config = strategy_section(
        evaluation_config, EvaluationStrategy.EMBEDDING_SIMILARITY
    )
    metric_name = str(strategy_config.get("metric", "cosine")).strip()
    metric_fn = registries.evaluations.get_metric(metric_name)
    embedding_config = client_section(
        strategy_config,
        evaluation_config,
        "embedding_client",
    )
    accessor = ConfigAccessor(
        strategy_config,
        prefix=f"evaluation.{EvaluationStrategy.EMBEDDING_SIMILARITY}",
        error_type=EvaluationConfigError,
    )
    return EmbeddingSimilarityEvaluator(
        embedding_client=ProviderBackedEmbeddingClient(
            registries.llm_providers.build_embedding_client(embedding_config)
        ),
        metric=metric_fn,
        attack_similarity_threshold=accessor.get_float(
            "attack_similarity_threshold", 0.75
        ),
        max_reference_similarity=accessor.get_float("max_reference_similarity", 0.6),
        attack_margin_threshold=accessor.get_float("attack_margin_threshold", 0.1),
    )


def build_llm_judge_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
) -> LLMJudgeEvaluator:
    strategy_config = strategy_section(evaluation_config, EvaluationStrategy.LLM_JUDGE)
    judge_config = client_section(strategy_config, evaluation_config, "judge_client")
    prompt_builder = resolve_judge_prompt_builder(registries, attack_id)
    return LLMJudgeEvaluator(
        judge_client=ProviderBackedJsonCompletionClient(
            registries.llm_providers.build_json_completion_client(
                judge_config,
                config_prefix="judge_client",
            )
        ),
        prompt_builder=prompt_builder,
    )


def _build_composite_evaluator(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
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
                registries=registries,
            )
        )
    return CompositeEvaluator(tuple(built), combine=combine)


def _build_evaluator_from_section(
    evaluation_config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
) -> Evaluator:
    strategy = parse_evaluation_strategy(evaluation_config)
    if strategy == EvaluationStrategy.COMPOSITE:
        raise EvaluationConfigError("nested composite evaluators are not supported")
    builder = registries.evaluations.get_evaluator_builder(strategy)
    return builder(evaluation_config, attack_id=attack_id, registries=registries)


def resolve_judge_prompt_builder(
    registries: ScanRegistries, attack_id: str
) -> JudgePromptBuilder:
    factory = registries.attack_methods.get(attack_id).judge_prompt_builder_factory
    if factory is None:
        supported = ", ".join(
            aid
            for aid in registries.attack_methods.ids()
            if registries.attack_methods.get(aid).judge_prompt_builder_factory
        )
        raise EvaluationConfigError(
            "llm_judge is not supported for this attack or judge prompts are missing; "
            f"attacks with judge support: {supported}"
        )
    return factory()


def evaluation_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "evaluation" not in config:
        return config
    evaluation = config["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise EvaluationConfigError("evaluation must be a mapping")
    return evaluation


def strategy_section(
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


def client_section(
    strategy_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    config = strategy_config.get(key, evaluation_config.get(key))
    if not isinstance(config, Mapping):
        raise EvaluationConfigError(f"evaluation.{key} must be configured")
    return config
