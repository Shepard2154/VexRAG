from collections.abc import Mapping, Sequence
from typing import Any

from vexrag.core.evaluation.errors import EvaluationDependencyError
from vexrag.core.llm.contracts import EmbeddingClient, JsonCompletionClient
from vexrag.core.llm.providers.errors import ProviderServiceError


class ProviderBackedEmbeddingClient:
    """Wraps a provider embedding client; maps provider failures to evaluation errors."""

    def __init__(self, inner: EmbeddingClient) -> None:
        self._inner = inner

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            return self._inner.embed_texts(texts)
        except ProviderServiceError as exc:
            raise EvaluationDependencyError(str(exc)) from exc


class ProviderBackedJsonCompletionClient:
    """Wraps a provider JSON client; maps provider failures to evaluation errors."""

    def __init__(self, inner: JsonCompletionClient) -> None:
        self._inner = inner

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]:
        try:
            return self._inner.complete_json(prompt)
        except ProviderServiceError as exc:
            raise EvaluationDependencyError(str(exc)) from exc
