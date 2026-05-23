from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvaluationStrategy(StrEnum):
    EMBEDDING_SIMILARITY = "embedding_similarity"
    LLM_JUDGE = "llm_judge"
    COMPOSITE = "composite"


class CombineMode(StrEnum):
    """Success criterion for composite evaluation with multiple evaluators."""

    ANY = "any"
    ALL = "all"


class JudgeAnswerLabel(StrEnum):
    """How an LLM judge classifies the target answer against expected outcomes."""

    CLEAN = "clean"
    ATTACK = "attack"
    UNRELATED = "unrelated"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class JudgeDetails:
    label: JudgeAnswerLabel
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured verdict returned by an evaluation strategy."""

    attack_successful: bool
    strategy: EvaluationStrategy | str
    completed: bool = True
    reason: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    judge: JudgeDetails | None = None
    children: tuple["EvaluationResult", ...] | None = None
    raw: str | Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()


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

    def __post_init__(self) -> None:
        for name in (
            "query",
            "actual_answer",
            "expected_clean_answer",
            "expected_attack_answer",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def context_lines(self) -> tuple[str, ...]:
        lines: list[str] = list(self.retrieved_contexts)
        if self.context_override is not None:
            lines.extend(self.context_override)
        return tuple(lines)


def attack_successful_from_judge_label(label: JudgeAnswerLabel) -> bool:
    return label == JudgeAnswerLabel.ATTACK


def make_incomplete_verdict(
    strategy: EvaluationStrategy | str,
    reason: str,
    *,
    raw: str | Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
) -> EvaluationResult:
    resolved_warnings = warnings if warnings else (reason,)
    return EvaluationResult(
        attack_successful=False,
        completed=False,
        strategy=strategy,
        reason=reason,
        raw=raw,
        warnings=resolved_warnings,
    )
