from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvaluationStrategy(StrEnum):
    EMBEDDING_SIMILARITY = "embedding_similarity"
    LLM_JUDGE = "llm_judge"


class CombineMode(StrEnum):
    """Success criterion for composite evaluation with multiple evaluators.

    Attributes:
        ANY: Attack succeeds if any completed child reports success.
             The overall evaluation is completed if at least one child completed.
        ALL: Attack succeeds only if every child completed and every child reports success.
             Otherwise, if any child is incomplete, the overall evaluation is incomplete.
    """

    ANY = "any"
    ALL = "all"


class JudgeAnswerLabel(StrEnum):
    """How an LLM judge classifies the target answer against expected outcomes.

    Attributes:
        CLEAN: Actual answer aligns with the benign reference.
        ATTACK: Actual answer aligns with the poisoned reference.
        UNRELATED: Actual answer does not clearly match either expected reference.
        INCONCLUSIVE: Insufficient evidence in the actual answer to choose clean or attack.
    """

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
    """Structured verdict returned by an evaluation strategy.

    - If ``completed`` is False, evaluation failed (provider error, parsing, etc.),
      then ``attack_successful`` is always False – do not score.
    - If ``completed`` is True, ``attack_successful`` indicates whether the attack
      succeeded, as determined by the concrete evaluator's logic.

    Attributes:
        attack_successful: True if attack succeeded, False otherwise.
        strategy: Evaluator ID.
        completed: False on fatal errors. Default True.
        reason: Human-readable explanation (logs/reports).
        scores: Numeric diagnostics (similarities, confidence, etc.).
        judge: Present for LLM judge when completed – normalized label and reason.
        children: Sub-verdicts for composite evaluators. None for leaf.
        raw: Unprocessed payload (JSON string/dict, or child raw map).
        warnings: Non‑fatal notices.
    """

    attack_successful: bool
    strategy: EvaluationStrategy | str
    completed: bool = True
    reason: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    judge: JudgeDetails | None = None
    children: tuple["EvaluationResult", ...] | None = None
    raw: str | Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()


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
