from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..contracts import CorpusPoisoningError


class FileTextPoisoner:
    """file_text corpus: one file per poison chunk."""

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
        root = self.path.resolve()
        for raw_document_id in document_ids:
            document_id = str(raw_document_id).strip()
            if not document_id:
                continue
            candidate = Path(document_id).expanduser()
            try:
                resolved = candidate.resolve()
            except OSError as exc:
                raise CorpusPoisoningError(
                    f"could not resolve corpus document path: {document_id}"
                ) from exc
            if resolved != root and root not in resolved.parents:
                raise CorpusPoisoningError(
                    f"refusing to delete path outside corpus directory: {document_id}"
                )
            try:
                resolved.unlink(missing_ok=True)
            except OSError as exc:
                raise CorpusPoisoningError(
                    f"could not delete poisoned text from file_text corpus: {resolved}"
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
