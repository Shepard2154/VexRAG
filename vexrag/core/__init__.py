from vexrag.core.evaluation import (
    CosineSimilarityMetric,
    EmbeddingClientProtocol,
    EvaluationInput,
    EvaluationResult,
    EvaluationStrategyProtocol,
    JudgeLLMProtocol,
    JudgePromptBuilderProtocol,
    JudgeResponse,
    JudgeResponseValidationError,
    LLMJudgeEvaluator,
    SemanticSimilarityEvaluator,
    SimilarityMetricProtocol,
    validate_judge_response,
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
    CorpusPoisoningAdapterProtocol,
    CorpusPoisoningError,
    FileTextCorpusPoisoningAdapter,
    RetrievalBackend,
)
from vexrag.core.scan import ScanVerdict
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
    # Evaluation types
    "EvaluationInput",
    "EvaluationResult",
    # Evaluation protocols
    "EmbeddingClientProtocol",
    "EvaluationStrategyProtocol",
    "JudgeLLMProtocol",
    "JudgePromptBuilderProtocol",
    "SimilarityMetricProtocol",
    # LLM judge evaluation
    "JudgeResponse",
    "JudgeResponseValidationError",
    "LLMJudgeEvaluator",
    "validate_judge_response",
    # Semantic similarity evaluation
    "CosineSimilarityMetric",
    "SemanticSimilarityEvaluator",
    # Providers
    "OllamaEmbeddingClient",
    "OllamaJudgeClient",
    "VLLMEmbeddingClient",
    "VLLMJudgeClient",
    "ProviderConfigError",
    "ProviderServiceError",
    "build_embedding_client",
    "build_judge_client",
    # Retrieval
    "CorpusPoisoningAdapterProtocol",
    "CorpusPoisoningError",
    "FileTextCorpusPoisoningAdapter",
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
