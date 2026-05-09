from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class CorpusPoisoningError(RuntimeError):
    """Corpus poisoning I/O failure."""


class CorpusPoisoningAdapterProtocol(Protocol):
    """Write poison chunks into retrieval storage."""

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]: ...

    def delete_texts(self, document_ids: Sequence[str]) -> None: ...
