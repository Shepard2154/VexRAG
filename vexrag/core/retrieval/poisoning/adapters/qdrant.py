import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.llm.contracts import EmbeddingClient

from ...errors import CorpusPoisoningError
from .._texts import nonempty_stripped_strs
from ..embeddings import embed_poison_vectors


def _qdrant_point_payload(text: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context": text,
        **{k: v for k, v in metadata.items() if k != "context"},
    }


def _build_qdrant_points(
    stripped: Sequence[str],
    vectors: Sequence[Sequence[float]],
    metadata: Mapping[str, Any],
    vector_name: str | None,
    qmodels: Any,
) -> tuple[list[Any], list[str]]:
    points: list[Any] = []
    ids: list[str] = []
    for text, vector in zip(stripped, vectors, strict=True):
        point_id = str(uuid.uuid4())
        ids.append(point_id)
        if vector_name:
            vec_payload: Any = {vector_name: vector}
        else:
            vec_payload = vector
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vec_payload,
                payload=_qdrant_point_payload(text, metadata),
            )
        )
    return points, ids


class QdrantPoisoner:
    """Qdrant upsert/delete for corpus poisoning."""

    def __init__(
        self,
        *,
        url: str | None,
        path: Path | None,
        collection: str,
        embedding_client: EmbeddingClient,
        vector_name: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
        l2_normalize: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise CorpusPoisoningError(
                "Qdrant corpus poisoning requires optional dependencies; "
                "install with: pip install 'vexrag[qdrant]'"
            ) from exc

        if url and path:
            raise CorpusPoisoningError(
                "qdrant corpus poisoning config must set only one of url or path"
            )
        if not url and not path:
            raise CorpusPoisoningError(
                "qdrant corpus poisoning requires qdrant.url or qdrant.path"
            )

        self._qmodels = qmodels
        self._collection = collection
        self._vector_name = vector_name or None
        self._owned_ids: set[str] = set()
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize

        if url:
            self._client = QdrantClient(
                url=url.strip(),
                timeout=timeout,
                api_key=api_key,
            )
        else:
            assert path is not None
            self._client = QdrantClient(path=str(path.resolve()), timeout=timeout)

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]:
        stripped = nonempty_stripped_strs(texts)
        if not stripped:
            return ()
        vectors = embed_poison_vectors(
            self._embedding_client,
            stripped,
            l2_normalize=self._l2_normalize,
        )
        points, ids = _build_qdrant_points(
            stripped,
            vectors,
            metadata,
            self._vector_name,
            self._qmodels,
        )
        for pid in ids:
            self._owned_ids.add(pid)
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
        except Exception as exc:
            for pid in ids:
                self._owned_ids.discard(pid)
            raise CorpusPoisoningError(f"qdrant upsert failed: {exc}") from exc
        return tuple(ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        to_delete = [
            did.strip()
            for did in document_ids
            if isinstance(did, str) and did.strip() and did.strip() in self._owned_ids
        ]
        if not to_delete:
            return
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=self._qmodels.PointIdsList(
                    points=list(to_delete),
                ),
            )
        except Exception as exc:
            raise CorpusPoisoningError(f"qdrant delete failed: {exc}") from exc
        for pid in to_delete:
            self._owned_ids.discard(pid)
