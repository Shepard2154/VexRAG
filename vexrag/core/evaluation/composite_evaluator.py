from collections.abc import Sequence
from typing import Any

from vexrag.core.evaluation.attack_verdict import (
    CombineMode,
    EvaluationResult,
    EvaluationStrategy,
)
from vexrag.core.evaluation.evaluator_protocols import Evaluator
from vexrag.core.evaluation.scan_case_input import EvaluationInput


def _child_prefix(index: int, strategy: EvaluationStrategy | str) -> str:
    return f"{index}:{strategy}"


def _merge_reason(
    sub: Sequence[EvaluationResult],
    *,
    if_empty: str | None = None,
) -> str | None:
    reasons = [result.reason for result in sub if result.reason]
    if reasons:
        return " | ".join(reasons)
    return if_empty


def _merge_warnings(sub: Sequence[EvaluationResult]) -> tuple[str, ...]:
    warnings: list[str] = []
    for index, result in enumerate(sub):
        prefix = _child_prefix(index, result.strategy)
        if not result.completed:
            warnings.append(f"[{prefix}] incomplete")
        for warning in result.warnings:
            warnings.append(f"[{prefix}] {warning}")
    return tuple(dict.fromkeys(warnings))


def _merge_scores(sub: Sequence[EvaluationResult]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, result in enumerate(sub):
        prefix = _child_prefix(index, result.strategy)
        for name, value in result.scores.items():
            scores[f"{prefix}:{name}"] = value
    return scores


def _merge_raw(sub: Sequence[EvaluationResult]) -> dict[str, Any] | None:
    raw_parts: dict[str, Any] = {}
    for index, result in enumerate(sub):
        if result.raw is not None:
            raw_parts[_child_prefix(index, result.strategy)] = result.raw
    return raw_parts or None


class CompositeEvaluator:
    """Runs several ``Evaluator`` implementations and merges verdicts."""

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        combine: CombineMode = CombineMode.ANY,
    ) -> None:
        if not evaluators:
            raise ValueError("evaluators must be non-empty")
        self._evaluators = tuple(evaluators)
        self._combine = combine

    @property
    def sub_evaluators(self) -> tuple[Evaluator, ...]:
        return self._evaluators

    @property
    def strategy(self) -> str:
        inner = "+".join(str(e.strategy) for e in self._evaluators)
        return f"composite({self._combine}:{inner})"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        sub = [ev.evaluate(evaluation_input) for ev in self._evaluators]

        if self._combine == CombineMode.ALL:
            if not all(r.completed for r in sub):
                return self._result(sub, completed=False, attack_successful=False)
            return self._result(
                sub,
                completed=True,
                attack_successful=all(r.attack_successful for r in sub),
            )

        if any(r.completed and r.attack_successful for r in sub):
            return self._result(sub, completed=True, attack_successful=True)
        if not any(r.completed for r in sub):
            return self._result(sub, completed=False, attack_successful=False)
        return self._result(sub, completed=True, attack_successful=False)

    def _result(
        self,
        sub: list[EvaluationResult],
        *,
        completed: bool,
        attack_successful: bool,
    ) -> EvaluationResult:
        reason_if_empty = (
            None if completed else "One or more evaluators did not complete"
        )
        return EvaluationResult(
            attack_successful=attack_successful,
            completed=completed,
            strategy=self.strategy,
            scores=_merge_scores(sub) if completed else {},
            reason=_merge_reason(sub, if_empty=reason_if_empty),
            warnings=_merge_warnings(sub),
            raw=_merge_raw(sub) if completed else None,
            children=tuple(sub),
        )
