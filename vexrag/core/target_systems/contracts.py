from typing import Protocol

from vexrag.core.target_systems.types import TargetSystemQuery, TargetSystemResponse


class TargetSystemAdapter(Protocol):
    """Adapter contract for invoking a target RAG system."""

    def answer(self, request: TargetSystemQuery) -> TargetSystemResponse: ...
