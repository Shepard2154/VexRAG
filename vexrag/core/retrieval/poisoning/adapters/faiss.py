import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.llm.contracts import EmbeddingClient

from ...errors import CorpusPoisoningError
from .._texts import nonempty_stripped_strs
from ..embeddings import embed_poison_vectors


def _ordered_ids_from_metadata(raw_meta: Mapping[str, Any]) -> list[int]:
    ordered_ids_raw = raw_meta.get("ordered_ids")
    if not isinstance(ordered_ids_raw, list):
        raise CorpusPoisoningError("metadata.json must contain an ordered_ids list")
    ordered_ids: list[int] = []
    for item in ordered_ids_raw:
        try:
            ordered_ids.append(int(item))
        except (TypeError, ValueError) as exc:
            raise CorpusPoisoningError("metadata ordered_ids must be integers") from exc
    return ordered_ids


def _assert_ntotal_matches(index: Any, ordered_ids: Sequence[int]) -> None:
    ntotal = int(index.ntotal)
    if ntotal != len(ordered_ids):
        raise CorpusPoisoningError(
            f"faiss index.ntotal ({ntotal}) does not match len(ordered_ids) "
            f"({len(ordered_ids)})"
        )


def _read_faiss_corpus_state(
    index_path: Path,
    metadata_path: Path,
    faiss: Any,
) -> tuple[Any, list[int]]:
    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise CorpusPoisoningError(f"could not read faiss index: {exc}") from exc
    try:
        raw_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusPoisoningError(
            f"could not read faiss metadata.json: {exc}"
        ) from exc
    ordered_ids = _ordered_ids_from_metadata(raw_meta)
    _assert_ntotal_matches(index, ordered_ids)
    return index, ordered_ids


def _persist_faiss_corpus(
    index_path: Path,
    metadata_path: Path,
    index: Any,
    ordered_ids: list[int],
    faiss: Any,
    *,
    os_error_message: str,
) -> None:
    try:
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(
            json.dumps({"ordered_ids": ordered_ids}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise CorpusPoisoningError(os_error_message) from exc


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
    for row_idx, sid in enumerate(ordered_ids):
        if sid in remove_int:
            continue
        keep_vectors.append(
            np.array(index.reconstruct(row_idx), dtype=np.float32).reshape(1, -1)
        )
        keep_ids.append(sid)
    if len(keep_vectors) == 0:
        new_index = faiss.IndexFlatIP(dim)
    else:
        stacked = np.vstack(keep_vectors)
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(stacked)
    return new_index, keep_ids


class FaissPoisoner:
    """FAISS IndexFlatIP corpus append/delete (index.faiss + metadata.json)."""

    def __init__(
        self,
        faiss_dir: Path,
        embedding_client: EmbeddingClient,
        *,
        l2_normalize: bool = False,
        poison_id_start: int = -1,
    ) -> None:
        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np
        except ImportError as exc:
            raise CorpusPoisoningError(
                "FAISS corpus poisoning requires optional dependencies; "
                "install with: pip install 'vexrag[faiss]'"
            ) from exc

        self._dir = faiss_dir
        self._index_path = faiss_dir / "index.faiss"
        self._metadata_path = faiss_dir / "metadata.json"
        self._embedding_client = embedding_client
        self._l2_normalize = l2_normalize
        self._owned_ids: set[str] = set()
        self._next_poison_id = poison_id_start
        self._faiss = faiss
        self._np = np

    def _require_corpus_files(self, *, detail_path: bool) -> None:
        if self._index_path.is_file() and self._metadata_path.is_file():
            return
        if detail_path:
            raise CorpusPoisoningError(
                f"faiss corpus directory must contain index.faiss and metadata.json: "
                f"{self._dir}"
            )
        raise CorpusPoisoningError("faiss corpus files are missing")

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
        np = self._np
        faiss = self._faiss

        self._require_corpus_files(detail_path=True)
        index, ordered_ids = _read_faiss_corpus_state(
            self._index_path, self._metadata_path, faiss
        )

        dim = int(index.d)
        addition = np.array(vectors, dtype=np.float32)
        if addition.ndim != 2 or addition.shape[1] != dim:
            raise CorpusPoisoningError(
                f"embedding dimension {addition.shape[1] if addition.ndim == 2 else 'n/a'} "
                f"does not match faiss index dimension {dim}"
            )

        new_ids: list[str] = []
        id_ints: list[int] = []
        for _ in stripped:
            self._next_poison_id -= 1
            pid_int = self._next_poison_id
            id_ints.append(pid_int)
            sid = str(pid_int)
            new_ids.append(sid)
            self._owned_ids.add(sid)

        try:
            index.add(addition)
        except Exception as exc:
            for sid in new_ids:
                self._owned_ids.discard(sid)
            raise CorpusPoisoningError(f"faiss index.add failed: {exc}") from exc

        ordered_ids.extend(id_ints)
        _persist_faiss_corpus(
            self._index_path,
            self._metadata_path,
            index,
            ordered_ids,
            faiss,
            os_error_message="could not persist faiss index",
        )
        return tuple(new_ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        to_remove = {
            did.strip()
            for did in document_ids
            if isinstance(did, str) and did.strip() and did.strip() in self._owned_ids
        }
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

        for sid in to_remove:
            self._owned_ids.discard(sid)
