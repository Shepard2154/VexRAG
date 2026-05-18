from collections.abc import Sequence
from dataclasses import dataclass

from vexrag.core.evaluation import EmbeddingClient

from .embedding_vectors import embed_text_vectors
from .text_normalization import normalized_nonempty_texts


@dataclass(frozen=True)
class EmbeddedTextBatch:
    texts: tuple[str, ...]
    vectors: list[list[float]]


def build_embedded_text_batch(
    embedding_client: EmbeddingClient,
    texts: Sequence[str],
    *,
    l2_normalize: bool,
) -> EmbeddedTextBatch | None:
    stripped_texts = tuple(normalized_nonempty_texts(texts))
    if not stripped_texts:
        return None
    vectors = embed_text_vectors(
        embedding_client,
        list(stripped_texts),
        l2_normalize=l2_normalize,
    )
    return EmbeddedTextBatch(texts=stripped_texts, vectors=vectors)
