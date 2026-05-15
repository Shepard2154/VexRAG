from vexrag.core.evaluation import EmbeddingClient
from vexrag.core.retrieval.poisoning.contracts import CorpusPoisoningError


def optional_l2_normalize_batch(
    vectors: list[list[float]],
    *,
    enabled: bool,
) -> list[list[float]]:
    if not enabled or not vectors:
        return vectors
    try:
        import numpy as np
    except ImportError as exc:
        raise CorpusPoisoningError(
            "l2_normalize for corpus poisoning requires numpy "
            "(install faiss/chroma/qdrant extra or pip install numpy)"
        ) from exc
    out: list[list[float]] = []
    for row in vectors:
        arr = np.array(row, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        out.append(arr.tolist())
    return out


def embed_poison_vectors(
    embedding_client: EmbeddingClient,
    texts: list[str],
    *,
    l2_normalize: bool,
) -> list[list[float]]:
    if not texts:
        return []
    raw = embedding_client.embed_texts(texts)
    vectors = [[float(x) for x in row] for row in raw]
    if len(vectors) != len(texts):
        raise CorpusPoisoningError(
            "embedding_client returned a different number of vectors than input texts"
        )
    return optional_l2_normalize_batch(vectors, enabled=l2_normalize)
