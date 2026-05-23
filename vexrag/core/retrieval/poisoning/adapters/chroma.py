import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.llm.contracts import EmbeddingClient

from ...errors import CorpusPoisoningError
from .._texts import nonempty_stripped_strs
from ..embeddings import embed_poison_vectors


def _chroma_document_metadatas(
    documents: Sequence[str],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in documents:
        row: dict[str, Any] = {"context": text}
        for k, v in metadata.items():
            if k == "context":
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
            else:
                row[k] = str(v)
        rows.append(row)
    return rows


class ChromaPoisoner:
    """Chroma collection add/delete for poisoning."""

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
        except ImportError as exc:
            raise CorpusPoisoningError(
                "Chroma corpus poisoning requires optional dependencies; "
                "install with: pip install 'vexrag[chroma]'"
            ) from exc

        if host and persist_directory:
            raise CorpusPoisoningError(
                "chroma config must set either persist_directory/path or host, not both"
            )
        if not host and not persist_directory:
            raise CorpusPoisoningError(
                "chroma corpus poisoning requires chroma.persist_directory "
                "or chroma.host"
            )

        if host:
            client = chromadb.HttpClient(host=host.strip(), port=port)
        else:
            assert persist_directory is not None
            client = chromadb.PersistentClient(path=str(persist_directory.resolve()))

        self._collection = client.get_or_create_collection(
            name=collection_name,
        )
        self._owned_ids: set[str] = set()
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize

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
        ids = [str(uuid.uuid4()) for _ in stripped]
        metadatas = _chroma_document_metadatas(stripped, metadata)
        for pid in ids:
            self._owned_ids.add(pid)
        try:
            self._collection.add(
                ids=ids,
                embeddings=vectors,
                documents=stripped,
                metadatas=metadatas,
            )
        except Exception as exc:
            for pid in ids:
                self._owned_ids.discard(pid)
            raise CorpusPoisoningError(f"chroma add failed: {exc}") from exc
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
            self._collection.delete(ids=to_delete)
        except Exception as exc:
            raise CorpusPoisoningError(f"chroma delete failed: {exc}") from exc
        for pid in to_delete:
            self._owned_ids.discard(pid)
