from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    """Attack-agnostic input for evaluating a target-system answer."""
    query: str
    actual_answer: str
    expected_clean_answer: str
    expected_attack_answer: str
    retrieved_contexts: tuple[str, ...] = ()
    context_override: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured verdict returned by an evaluation strategy."""

    attack_successful: bool
    strategy: str
    scores: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None
    raw_response: str | Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluation_completed: bool = True


class EvaluationStrategyProtocol(Protocol):
    """Evaluation strategy contract shared by attack implementations."""
    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult: ...


class EmbeddingClientProtocol(Protocol):
    """Client contract for embedding text batches."""
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SimilarityMetricProtocol(Protocol):
    """Metric contract for comparing two embedding vectors."""
    def score(self, left: Sequence[float], right: Sequence[float]) -> float: ...


class JudgeLLMProtocol(Protocol):
    """Client contract for JSON-producing LLM judge calls."""
    def complete_json(self, prompt: str) -> str | Mapping[str, Any]: ...


class JudgePromptBuilderProtocol(Protocol):
    """Builds attack-specific prompts for a generic LLM judge strategy."""
    def build_prompt(self, evaluation_input: EvaluationInput) -> str: ...
