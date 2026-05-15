from vexrag.core.contracts import ScanVerdict, TargetStyle
from vexrag.core.correct_answer_prompt import build_correct_answer_prompt
from vexrag.core.evaluation import (
    CompositeEvaluator,
    EmbeddingClient,
    EmbeddingSimilarityEvaluator,
    EvaluationInput,
    EvaluationResult,
    Evaluator,
    JudgeClient,
    JudgeDetails,
    JudgePromptBuilder,
    JudgeResponseValidationError,
    LLMJudgeEvaluator,
    parse_judge_llm_response,
)
from vexrag.core.llm_response_validation import (
    LLMPayloadValidationError,
    coerce_payload_to_dict,
    validate_correct_answer_payload,
)
from vexrag.core.providers import (
    OllamaEmbeddingClient,
    OllamaJudgeClient,
    ProviderConfigError,
    ProviderServiceError,
    VLLMEmbeddingClient,
    VLLMJudgeClient,
    build_embedding_client,
    build_judge_client,
)
from vexrag.core.retrieval import (
    ChromaPoisoner,
    CorpusPoisoningAdapterProtocol,
    CorpusPoisoningError,
    FaissPoisoner,
    FileTextPoisoner,
    QdrantPoisoner,
    RetrievalBackend,
)
from vexrag.core.target import (
    HTTPResponsePaths,
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
    HTTPTargetSystemAdapterError,
    TargetSystemAdapterProtocol,
    TargetSystemQuery,
    TargetSystemResponse,
)

__all__ = [
    # Shared attack typing
    "TargetStyle",
    # LLM JSON helpers
    "LLMPayloadValidationError",
    "build_correct_answer_prompt",
    "coerce_payload_to_dict",
    "validate_correct_answer_payload",
    # Evaluation types
    "EvaluationInput",
    "EvaluationResult",
    # Evaluation protocols
    "Evaluator",
    "EmbeddingClient",
    "JudgeClient",
    "JudgePromptBuilder",
    # LLM judge evaluation
    "JudgeDetails",
    "JudgeResponseValidationError",
    "LLMJudgeEvaluator",
    "parse_judge_llm_response",
    # Embedding similarity evaluation
    "EmbeddingSimilarityEvaluator",
    "CompositeEvaluator",
    # Providers
    "OllamaEmbeddingClient",
    "OllamaJudgeClient",
    "VLLMEmbeddingClient",
    "VLLMJudgeClient",
    "ProviderConfigError",
    "ProviderServiceError",
    "build_embedding_client",
    "build_judge_client",
    # Corpus poisoning
    "ChromaPoisoner",
    "FaissPoisoner",
    "QdrantPoisoner",
    # Retrieval
    "CorpusPoisoningAdapterProtocol",
    "CorpusPoisoningError",
    "FileTextPoisoner",
    "RetrievalBackend",
    # Scan
    "ScanVerdict",
    # Target system
    "HTTPResponsePaths",
    "HTTPTargetSystemAdapter",
    "HTTPTargetSystemAdapterConfig",
    "HTTPTargetSystemAdapterError",
    "TargetSystemAdapterProtocol",
    "TargetSystemQuery",
    "TargetSystemResponse",
]
