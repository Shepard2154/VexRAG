from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class RetrievalCorpusAdapter(Protocol):
    """Write adversarial chunks into a retrieval corpus and clean them up."""

    def add_texts(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[str, ...]: ...

    def delete_texts(self, document_ids: Sequence[str]) -> None: ...
