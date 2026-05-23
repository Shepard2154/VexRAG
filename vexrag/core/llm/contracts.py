from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class JsonCompletionClient(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]: ...
