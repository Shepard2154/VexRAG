from collections.abc import Mapping, Sequence
from typing import Any

from vexrag.core.evaluation.errors import EvaluationDependencyError
from vexrag.core.evaluation.evaluator_protocols import EmbeddingClient, JudgeClient
from vexrag.core.providers.errors import ProviderServiceError


class ProviderBackedEmbeddingClient:
    """Wraps a provider embedding client; maps provider failures to evaluation errors."""

    def __init__(self, inner: EmbeddingClient) -> None:
        self._inner = inner

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            return self._inner.embed_texts(texts)
        except ProviderServiceError as exc:
            raise EvaluationDependencyError(str(exc)) from exc


class ProviderBackedJudgeClient:
    """Wraps a provider judge client; maps provider failures to evaluation errors."""

    def __init__(self, inner: JudgeClient) -> None:
        self._inner = inner

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]:
        try:
            return self._inner.complete_json(prompt)
        except ProviderServiceError as exc:
            raise EvaluationDependencyError(str(exc)) from exc
