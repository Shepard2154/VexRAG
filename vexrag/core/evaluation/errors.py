from vexrag.core.exceptions import VexRAGCoreError


class EvaluatorError(VexRAGCoreError):
    """Base exception for evaluation-layer failures."""


class EvaluationDependencyError(EvaluatorError):
    """Embedding or judge API unavailable or returned an invalid response."""


class EmbeddingResponseError(EvaluationDependencyError):
    """Embedding client response does not match the expected contract."""


class JudgeResponseValidationError(EvaluatorError):
    """LLM judge response does not match the expected schema."""


class EmbeddingVectorError(EvaluatorError):
    """Invalid embedding vectors passed to a similarity metric."""


class EmptyEmbeddingVectorError(EmbeddingVectorError):
    """Embedding vector has no dimensions."""


class EmbeddingDimensionMismatchError(EmbeddingVectorError):
    """Embedding vectors have different dimensions."""


class ZeroNormEmbeddingError(EmbeddingVectorError):
    """Embedding vector has zero norm."""
