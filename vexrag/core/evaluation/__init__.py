from vexrag.core.evaluation.adapters import (
    ProviderBackedEmbeddingClient,
    ProviderBackedJsonCompletionClient,
)
from vexrag.core.evaluation.contracts import Evaluator, JudgePromptBuilder
from vexrag.core.evaluation.errors import (
    EmbeddingDimensionMismatchError,
    EmbeddingResponseError,
    EmbeddingVectorError,
    EmptyEmbeddingVectorError,
    EvaluationDependencyError,
    EvaluatorError,
    JudgeResponseValidationError,
    ZeroNormEmbeddingError,
)
from vexrag.core.evaluation.parsing import parse_judge_llm_response
from vexrag.core.evaluation.registry import EvaluationRegistry
from vexrag.core.evaluation.strategies import (
    CompositeEvaluator,
    EmbeddingSimilarityEvaluator,
    LLMJudgeEvaluator,
)
from vexrag.core.evaluation.types import (
    CombineMode,
    EvaluationInput,
    EvaluationResult,
    EvaluationStrategy,
    JudgeAnswerLabel,
    JudgeDetails,
    attack_successful_from_judge_label,
    make_incomplete_verdict,
)
from vexrag.core.llm.contracts import EmbeddingClient, JsonCompletionClient

__all__ = [
    # Verdict types
    "CombineMode",
    "EvaluationResult",
    "EvaluationStrategy",
    "JudgeAnswerLabel",
    "JudgeDetails",
    "attack_successful_from_judge_label",
    "make_incomplete_verdict",
    # Scan input
    "EvaluationInput",
    # Protocols
    "Evaluator",
    "EmbeddingClient",
    "JsonCompletionClient",
    "JudgePromptBuilder",
    "EvaluationRegistry",
    # Implementations
    "EmbeddingSimilarityEvaluator",
    "LLMJudgeEvaluator",
    "CompositeEvaluator",
    "parse_judge_llm_response",
    # Provider adapters (manual composition)
    "ProviderBackedEmbeddingClient",
    "ProviderBackedJsonCompletionClient",
    # Errors
    "EvaluatorError",
    "EvaluationDependencyError",
    "EmbeddingResponseError",
    "JudgeResponseValidationError",
    "EmbeddingVectorError",
    "EmptyEmbeddingVectorError",
    "EmbeddingDimensionMismatchError",
    "ZeroNormEmbeddingError",
]
