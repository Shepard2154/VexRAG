from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

HTTPMethod = Literal["GET", "POST", "PUT", "PATCH"]


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


@dataclass(frozen=True, slots=True)
class HTTPResponsePaths:
    """JSON paths used to map a target-system response into core contracts."""

    answer: str = "answer"
    contexts: str | None = "contexts"


@dataclass(frozen=True, slots=True)
class HTTPTargetSystemAdapterConfig:
    """Configuration for invoking a JSON-over-HTTP target RAG system."""

    base_url: str
    route: str = ""
    method: HTTPMethod | str = "POST"
    timeout: float | None = 10.0
    request_template: Mapping[str, Any] = field(
        default_factory=lambda: {
            "query": "{query}",
            "contexts": "{contexts}",
        }
    )
    response_paths: HTTPResponsePaths | Mapping[str, str | None] = field(
        default_factory=HTTPResponsePaths
    )
    headers: Mapping[str, str] = field(default_factory=dict)
    include_raw_response_in_metadata: bool = False

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        method = self.method.upper()
        object.__setattr__(self, "method", method)
        if method not in {"GET", "POST", "PUT", "PATCH"}:
            raise ValueError("method must be one of GET, POST, PUT, PATCH")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if isinstance(self.response_paths, Mapping):
            object.__setattr__(
                self,
                "response_paths",
                HTTPResponsePaths(
                    answer=str(self.response_paths.get("answer", "answer")),
                    contexts=self._optional_path(
                        self.response_paths.get("contexts", "contexts")
                    ),
                ),
            )

    @staticmethod
    def _optional_path(value: str | None) -> str | None:
        return None if value is None else str(value)
