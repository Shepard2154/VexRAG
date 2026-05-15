from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from vexrag.core.evaluation.dto import EvaluationInput, EvaluationResult


class EvaluationStrategyProtocol(Protocol):
    """Evaluation strategy contract shared by attack implementations."""

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult: ...


class EmbeddingClientProtocol(Protocol):
    """Client contract for embedding text batches."""

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class JudgeLLMProtocol(Protocol):
    """Client contract for JSON-producing LLM judge calls."""

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]: ...


class JudgePromptBuilderProtocol(Protocol):
    """Builds attack-specific prompts for a generic LLM judge strategy."""

    def build_prompt(self, evaluation_input: EvaluationInput) -> str: ...
