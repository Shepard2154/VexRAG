from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vexrag.core.retrieval.contracts import CorpusPoisoner

RetrievalBackendBuilder = Callable[[Mapping[str, Any]], CorpusPoisoner | None]


@dataclass(frozen=True, slots=True)
class RetrievalBackendRegistry:
    _builders: Mapping[str, RetrievalBackendBuilder]

    def get(self, backend: str) -> RetrievalBackendBuilder:
        key = backend.strip()
        try:
            return self._builders[key]
        except KeyError as err:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(
                f"unknown retrieval backend {backend!r}; supported: {supported}"
            ) from err
