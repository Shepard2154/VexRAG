import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    RetrievalCorpusPersistenceError,
)


def _ordered_ids_from_metadata(raw_meta: Mapping[str, Any]) -> list[int]:
    ordered_ids_raw = raw_meta.get("ordered_ids")
    if not isinstance(ordered_ids_raw, list):
        raise RetrievalCorpusError("metadata.json must contain an ordered_ids list")
    ordered_ids: list[int] = []
    for item in ordered_ids_raw:
        try:
            ordered_ids.append(int(item))
        except (TypeError, ValueError) as exc:
            raise RetrievalCorpusError("metadata ordered_ids must be integers") from exc
    return ordered_ids


def _assert_ntotal_matches(index: Any, ordered_ids: Sequence[int]) -> None:
    ntotal = int(index.ntotal)
    if ntotal != len(ordered_ids):
        raise RetrievalCorpusError(
            f"faiss index.ntotal ({ntotal}) does not match len(ordered_ids) "
            f"({len(ordered_ids)})"
        )


def _assert_supported_index(index: Any, faiss: Any) -> None:
    index_flat_ip = getattr(faiss, "IndexFlatIP", None)
    if index_flat_ip is not None and isinstance(index, index_flat_ip):
        return
    raise RetrievalCorpusError("FAISS corpus adapter only supports IndexFlatIP")


def _read_faiss_corpus_state(
    index_path: Path,
    metadata_path: Path,
    faiss: Any,
) -> tuple[Any, list[int]]:
    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError as exc:
        raise RetrievalCorpusPersistenceError(
            f"could not read faiss index: {exc}"
        ) from exc
    try:
        raw_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalCorpusPersistenceError(
            f"could not read faiss metadata.json: {exc}"
        ) from exc
    ordered_ids = _ordered_ids_from_metadata(raw_meta)
    _assert_ntotal_matches(index, ordered_ids)
    _assert_supported_index(index, faiss)
    return index, ordered_ids


def _sibling_temp_path(path: Path, label: str) -> Path:
    return path.with_name(f".{path.name}.{label}-{uuid4().hex}")


def _persist_faiss_corpus(
    index_path: Path,
    metadata_path: Path,
    index: Any,
    ordered_ids: list[int],
    faiss: Any,
    *,
    os_error_message: str,
) -> None:
    index_tmp_path = _sibling_temp_path(index_path, "tmp")
    metadata_tmp_path = _sibling_temp_path(metadata_path, "tmp")
    index_backup_path = _sibling_temp_path(index_path, "backup")
    metadata_backup_path = _sibling_temp_path(metadata_path, "backup")
    committed = False
    try:
        faiss.write_index(index, str(index_tmp_path))
        metadata_tmp_path.write_text(
            json.dumps({"ordered_ids": ordered_ids}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index_path.replace(index_backup_path)
        metadata_path.replace(metadata_backup_path)
        index_tmp_path.replace(index_path)
        metadata_tmp_path.replace(metadata_path)
        committed = True
    except (OSError, RuntimeError) as exc:
        with suppress(OSError):
            if index_backup_path.exists():
                index_backup_path.replace(index_path)
        with suppress(OSError):
            if metadata_backup_path.exists():
                metadata_backup_path.replace(metadata_path)
        raise RetrievalCorpusPersistenceError(os_error_message) from exc
    finally:
        for path in (index_tmp_path, metadata_tmp_path):
            with suppress(OSError):
                path.unlink()
        if committed:
            for path in (index_backup_path, metadata_backup_path):
                with suppress(OSError):
                    path.unlink()


def _allocate_added_point_ids(
    ordered_ids: Sequence[int],
    count: int,
    *,
    point_id_start: int,
) -> list[int]:
    next_point_id = min(min(ordered_ids, default=0) - 1, point_id_start)
    return [next_point_id - offset for offset in range(count)]


def _rebuild_index_without_point_ids(
    index: Any,
    ordered_ids: list[int],
    remove_int: set[int],
    faiss: Any,
    np: Any,
) -> tuple[Any, list[int]]:
    dim = int(index.d)
    keep_vectors: list[Any] = []
    keep_ids: list[int] = []
    try:
        for row_idx, sid in enumerate(ordered_ids):
            if sid in remove_int:
                continue
            keep_vectors.append(
                np.array(index.reconstruct(row_idx), dtype=np.float32).reshape(1, -1)
            )
            keep_ids.append(sid)
    except RuntimeError as exc:
        raise RetrievalCorpusBackendError(
            f"faiss index reconstruct failed: {exc}"
        ) from exc
    new_index = faiss.IndexFlatIP(dim)
    if keep_vectors:
        try:
            new_index.add(np.vstack(keep_vectors))
        except RuntimeError as exc:
            raise RetrievalCorpusBackendError(
                f"faiss index rebuild failed: {exc}"
            ) from exc
    return new_index, keep_ids


class FaissCorpusAdapter:
    """FAISS IndexFlatIP corpus append/delete adapter."""

    def __init__(
        self,
        faiss_dir: Path,
        embedding_client: EmbeddingClient,
        *,
        l2_normalize: bool = False,
        added_point_id_start: int = -1,
    ) -> None:
        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np
        except ImportError as exc:
            raise RetrievalCorpusDependencyError(
                "FAISS retrieval corpus writes require optional dependencies; "
                "install with: pip install 'vexrag[faiss]'"
            ) from exc

        self._dir = faiss_dir
        self._index_path = faiss_dir / "index.faiss"
        self._metadata_path = faiss_dir / "metadata.json"
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize
        self._created_document_ids: set[str] = set()
        self._added_point_id_start = added_point_id_start
        self._faiss = faiss
        self._np = np

    def _require_corpus_files(self, *, detail_path: bool) -> None:
        if self._index_path.is_file() and self._metadata_path.is_file():
            return
        if detail_path:
            raise RetrievalCorpusError(
                f"faiss corpus directory must contain index.faiss and metadata.json: "
                f"{self._dir}"
            )
        raise RetrievalCorpusError("faiss corpus files are missing")

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
        np = self._np
        faiss = self._faiss

        self._require_corpus_files(detail_path=True)
        index, ordered_ids = _read_faiss_corpus_state(
            self._index_path, self._metadata_path, faiss
        )

        dim = int(index.d)
        addition = np.array(batch.vectors, dtype=np.float32)
        if addition.ndim != 2 or addition.shape[1] != dim:
            raise RetrievalCorpusError(
                f"embedding dimension {addition.shape[1] if addition.ndim == 2 else 'n/a'} "
                f"does not match faiss index dimension {dim}"
            )

        id_ints = _allocate_added_point_ids(
            ordered_ids,
            len(batch.texts),
            point_id_start=self._added_point_id_start,
        )
        new_ids = [str(pid_int) for pid_int in id_ints]
        remembered_ids = remember_created_document_ids(
            self._created_document_ids, new_ids
        )

        try:
            index.add(addition)
        except RuntimeError as exc:
            forget_created_document_ids(self._created_document_ids, remembered_ids)
            raise RetrievalCorpusBackendError(f"faiss index.add failed: {exc}") from exc

        ordered_ids.extend(id_ints)
        try:
            _persist_faiss_corpus(
                self._index_path,
                self._metadata_path,
                index,
                ordered_ids,
                faiss,
                os_error_message="could not persist faiss index",
            )
        except RetrievalCorpusError:
            forget_created_document_ids(self._created_document_ids, remembered_ids)
            raise
        return tuple(new_ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        to_remove = created_document_ids_for_cleanup(
            document_ids, self._created_document_ids
        )
        if not to_remove:
            return

        faiss = self._faiss
        np = self._np

        self._require_corpus_files(detail_path=False)
        index, ordered_ids = _read_faiss_corpus_state(
            self._index_path, self._metadata_path, faiss
        )

        remove_int = {int(x) for x in to_remove}
        new_index, keep_ids = _rebuild_index_without_point_ids(
            index,
            ordered_ids,
            remove_int,
            faiss,
            np,
        )
        _persist_faiss_corpus(
            self._index_path,
            self._metadata_path,
            new_index,
            keep_ids,
            faiss,
            os_error_message="could not persist faiss index after delete",
        )

        forget_created_document_ids(self._created_document_ids, to_remove)
