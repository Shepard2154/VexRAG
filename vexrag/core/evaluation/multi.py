from collections.abc import Sequence
from typing import Any

from vexrag.core.evaluation.protocols import (
    EvaluationInput,
    EvaluationResult,
    EvaluationStrategyProtocol,
)


class MultiEvaluator:
    """Runs several ``EvaluationStrategyProtocol`` implementations and merges verdicts."""

    def __init__(
        self,
        evaluators: Sequence[EvaluationStrategyProtocol],
        *,
        combine: str = "any",
    ) -> None:
        if not evaluators:
            raise ValueError("evaluators must be non-empty")
        self._evaluators = tuple(evaluators)
        if combine not in ("any", "all"):
            raise ValueError("combine must be 'any' or 'all'")
        self._combine: str = combine

    @property
    def sub_evaluators(self) -> tuple[EvaluationStrategyProtocol, ...]:
        return self._evaluators

    @property
    def strategy(self) -> str:
        inner = "+".join(e.strategy for e in self._evaluators)
        return f"multi({self._combine}:{inner})"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        sub: list[EvaluationResult] = [
            ev.evaluate(evaluation_input) for ev in self._evaluators
        ]
        if self._combine == "all":
            attack_ok = all(r.attack_successful for r in sub)
        else:
            attack_ok = any(r.attack_successful for r in sub)

        scores: dict[str, float] = {}
        for index, result in enumerate(sub):
            prefix = f"{index}:{result.strategy}"
            for name, value in result.scores.items():
                scores[f"{prefix}:{name}"] = value

        reasons = [r.reason for r in sub if r.reason]
        reason = " | ".join(reasons) if reasons else None

        warnings: list[str] = []
        for index, result in enumerate(sub):
            for warning in result.warnings:
                warnings.append(f"[{index}:{result.strategy}] {warning}")
        raw_parts: dict[str, Any] = {}
        for index, result in enumerate(sub):
            if result.raw_response is not None:
                raw_parts[f"{index}:{result.strategy}"] = result.raw_response

        metadata: dict[str, Any] = {
            "combine": self._combine,
            "sub_results": tuple(
                {
                    "strategy": r.strategy,
                    "attack_successful": r.attack_successful,
                    "scores": dict(r.scores),
                    "reason": r.reason,
                }
                for r in sub
            ),
        }

        return EvaluationResult(
            attack_successful=attack_ok,
            strategy=self.strategy,
            scores=scores,
            reason=reason,
            raw_response=raw_parts if raw_parts else None,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata=metadata,
        )
