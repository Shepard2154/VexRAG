from collections.abc import Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol


class RetrievalBackend(StrEnum):
    """backend used to hold retrieved context."""

    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"
    FILE_TEXT = "file_text"

    def uses_named_collection(self) -> bool:
        """True when the backend uses a collection name."""
        return self is not RetrievalBackend.FILE_TEXT


class CorpusPoisoningError(RuntimeError):
    """Raised when poisoned texts cannot be written to a retrieval corpus."""


class CorpusPoisoningAdapterProtocol(Protocol):
    """Adapter contract for writing poisoned texts into retrieval storage."""

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]: ...

    def delete_texts(self, document_ids: Sequence[str]) -> None: ...


class FileTextCorpusPoisoningAdapter:
    """Writes poisoned texts as standalone files in a file_text corpus."""

    __slots__ = ("path", "filename_prefix")

    def __init__(self, path: Path, filename_prefix: str) -> None:
        self.path = path
        self.filename_prefix = filename_prefix

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CorpusPoisoningError(
                f"could not create file_text corpus path: {self.path}"
            ) from exc

        document_ids: list[str] = []
        for index, text in enumerate(texts, start=1):
            stripped_text = text.strip()
            if not stripped_text:
                continue
            document_path = self.path / self._filename(stripped_text, metadata, index)
            try:
                document_path.write_text(stripped_text + "\n", encoding="utf-8")
            except OSError as exc:
                raise CorpusPoisoningError(
                    f"could not write poisoned text to file_text corpus: {document_path}"
                ) from exc
            document_ids.append(str(document_path))
        return tuple(document_ids)

    def delete_texts(self, document_ids: Sequence[str]) -> None:
        for raw_document_id in document_ids:
            document_id = str(raw_document_id).strip()
            if not document_id:
                continue
            document_path = Path(document_id)
            try:
                document_path.unlink(missing_ok=True)
            except OSError as exc:
                raise CorpusPoisoningError(
                    f"could not delete poisoned text from file_text corpus: {document_path}"
                ) from exc

    def _filename(
        self,
        text: str,
        metadata: Mapping[str, Any],
        index: int,
    ) -> str:
        case = _slug(metadata.get("case_id", metadata.get("case_index", "case")))
        run = _slug(metadata.get("run_index", "run"))
        adversarial_text_index = _slug(metadata.get("adversarial_text_index", index))
        digest = sha256(f"{metadata!r}\n{text}".encode()).hexdigest()[:12]
        return (
            f"{self.filename_prefix}_{case}_{run}_{adversarial_text_index}_{digest}.txt"
        )


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    return slug.strip("_") or "item"
