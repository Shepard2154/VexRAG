from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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
