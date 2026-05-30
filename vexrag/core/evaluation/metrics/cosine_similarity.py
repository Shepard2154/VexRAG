from collections.abc import Sequence
from math import sqrt

from vexrag.core.evaluation.errors import (
    EmbeddingDimensionMismatchError,
    EmptyEmbeddingVectorError,
    ZeroNormEmbeddingError,
)


def _dot_product(left_vector: Sequence[float], right_vector: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left_vector, right_vector, strict=True))


def _l2_norm(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _validate_vectors(vector_a: Sequence[float], vector_b: Sequence[float]) -> None:
    if not vector_a or not vector_b:
        raise EmptyEmbeddingVectorError("embedding vectors must not be empty")
    if len(vector_a) != len(vector_b):
        raise EmbeddingDimensionMismatchError(
            "embedding vectors must have the same length"
        )


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    vector_a = tuple(vector_a)
    vector_b = tuple(vector_b)
    _validate_vectors(vector_a, vector_b)

    dot_product = _dot_product(vector_a, vector_b)
    vector_a_norm = _l2_norm(vector_a)
    vector_b_norm = _l2_norm(vector_b)

    if vector_a_norm == 0 or vector_b_norm == 0:
        raise ZeroNormEmbeddingError("embedding vectors must not have zero magnitude")
    return dot_product / (vector_a_norm * vector_b_norm)
