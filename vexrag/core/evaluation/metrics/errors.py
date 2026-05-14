class EmbeddingVectorError(ValueError):
    """Invalid embedding vectors passed to a metric."""


class EmptyEmbeddingVectorError(EmbeddingVectorError):
    """Raised when an embedding vector has no dimensions."""


class EmbeddingDimensionMismatchError(EmbeddingVectorError):
    """Raised when embedding vectors have different dimensions."""


class ZeroNormEmbeddingError(EmbeddingVectorError):
    """Raised when an embedding vector has a zero norm."""
