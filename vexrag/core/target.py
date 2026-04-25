from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TargetSystemQuery:
    """Transport-neutral query sent to a target RAG system."""

    query: str
    contexts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TargetSystemResponse:
    """Transport-neutral response returned by a target RAG system."""

    answer: str
    contexts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TargetSystemAdapterProtocol(Protocol):
    """Adapter contract for invoking a target RAG system."""

    def answer(self, request: TargetSystemQuery) -> TargetSystemResponse: ...
