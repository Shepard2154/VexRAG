import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.evaluation import EmbeddingClient
from vexrag.core.retrieval.document_ids import (
    created_document_ids_for_cleanup,
    forget_created_document_ids,
    remember_created_document_ids,
)
from vexrag.core.retrieval.embedding_inputs import embedded_text_batch
from vexrag.core.retrieval.errors import (
    RetrievalCorpusBackendError,
    RetrievalCorpusDependencyError,
    RetrievalCorpusError,
)


def _chroma_document_metadatas(
    documents: Sequence[str],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in documents:
        row: dict[str, Any] = {"context": text}
        for k, v in metadata.items():
            if k == "context" or v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                row[k] = v
            else:
                row[k] = str(v)
        rows.append(row)
    return rows


class ChromaCorpusAdapter:
    """Chroma collection add/delete adapter for retrieval corpus writes."""

    def __init__(
        self,
        *,
        persist_directory: Path | None,
        host: str | None,
        port: int,
        collection_name: str,
        embedding_client: EmbeddingClient,
        l2_normalize: bool = False,
    ) -> None:
        try:
            import chromadb
            from chromadb.errors import (
                ChromaError,
                DuplicateIDError,
                InvalidCollectionException,
                InvalidDimensionException,
                NotFoundError,
            )
        except ImportError as exc:
            raise RetrievalCorpusDependencyError(
                "Chroma retrieval corpus writes require optional dependencies; "
                "install with: pip install 'vexrag[chroma]'"
            ) from exc

        if host and persist_directory:
            raise RetrievalCorpusError(
                "chroma config must set either persist_directory/path or host, not both"
            )
        if not host and not persist_directory:
            raise RetrievalCorpusError(
                "chroma retrieval config requires chroma.persist_directory or chroma.host"
            )

        self._chroma_backend_errors = (
            ChromaError,
            DuplicateIDError,
            InvalidCollectionException,
            InvalidDimensionException,
            NotFoundError,
        )
        if host:
            client = chromadb.HttpClient(host=host.strip(), port=port)
        else:
            assert persist_directory is not None
            client = chromadb.PersistentClient(path=str(persist_directory.resolve()))

        self._collection = client.get_or_create_collection(
            name=collection_name,
        )
        self._created_document_ids: set[str] = set()
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]:
        batch = embedded_text_batch(
            self._embedding_client,
            texts,
            l2_normalize=self._l2_normalize,
        )
        if batch is None:
            return ()
        ids = [str(uuid.uuid4()) for _ in batch.texts]
        metadatas = _chroma_document_metadatas(batch.texts, metadata)
        remembered_ids = remember_created_document_ids(self._created_document_ids, ids)
        try:
            self._collection.add(
                ids=ids,
                embeddings=batch.vectors,
                documents=list(batch.texts),
                metadatas=metadatas,
            )
        except self._chroma_backend_errors as exc:
            forget_created_document_ids(self._created_document_ids, remembered_ids)
            raise RetrievalCorpusBackendError(f"chroma add failed: {exc}") from exc
        return tuple(ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        to_delete = created_document_ids_for_cleanup(
            document_ids, self._created_document_ids
        )
        if not to_delete:
            return
        try:
            self._collection.delete(ids=to_delete)
        except self._chroma_backend_errors as exc:
            raise RetrievalCorpusBackendError(f"chroma delete failed: {exc}") from exc
        forget_created_document_ids(self._created_document_ids, to_delete)
