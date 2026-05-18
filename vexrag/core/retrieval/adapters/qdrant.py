import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.evaluation import EmbeddingClient
from vexrag.core.retrieval._internal.embedding_batches import build_embedded_text_batch
from vexrag.core.retrieval._internal.id_tracking import (
    collect_created_document_ids_for_cleanup,
    forget_created_document_ids,
    remember_created_document_ids,
)
from vexrag.core.retrieval.errors import (
    RetrievalCorpusBackendError,
    RetrievalCorpusDependencyError,
    RetrievalCorpusError,
)


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


class QdrantCorpusAdapter:
    """Qdrant upsert/delete adapter for retrieval corpus writes."""

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
            from qdrant_client.http.exceptions import (
                ApiException,
                ResponseHandlingException,
                UnexpectedResponse,
            )
        except ImportError as exc:
            raise RetrievalCorpusDependencyError(
                "Qdrant retrieval corpus writes require optional dependencies; "
                "install with: pip install 'vexrag[qdrant]'"
            ) from exc

        if url and path:
            raise RetrievalCorpusError(
                "qdrant retrieval config must set only one of url or path"
            )
        if not url and not path:
            raise RetrievalCorpusError(
                "qdrant retrieval config requires qdrant.url or qdrant.path"
            )

        self._qmodels = qmodels
        self._collection = collection
        self._vector_name = vector_name or None
        self._created_document_ids: set[str] = set()
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize
        self._qdrant_backend_errors = (
            ApiException,
            ResponseHandlingException,
            UnexpectedResponse,
        )

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
        batch = build_embedded_text_batch(
            self._embedding_client,
            texts,
            l2_normalize=self._l2_normalize,
        )
        if batch is None:
            return ()
        points, ids = _build_qdrant_points(
            batch.texts,
            batch.vectors,
            metadata,
            self._vector_name,
            self._qmodels,
        )
        remembered_ids = remember_created_document_ids(self._created_document_ids, ids)
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
        except self._qdrant_backend_errors as exc:
            forget_created_document_ids(self._created_document_ids, remembered_ids)
            raise RetrievalCorpusBackendError(f"qdrant upsert failed: {exc}") from exc
        return tuple(ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        to_delete = collect_created_document_ids_for_cleanup(
            document_ids, self._created_document_ids
        )
        if not to_delete:
            return
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=self._qmodels.PointIdsList(
                    points=list(to_delete),
                ),
            )
        except self._qdrant_backend_errors as exc:
            raise RetrievalCorpusBackendError(f"qdrant delete failed: {exc}") from exc
        forget_created_document_ids(self._created_document_ids, to_delete)
