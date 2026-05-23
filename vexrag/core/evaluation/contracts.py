from typing import Protocol

from vexrag.core.evaluation.types import EvaluationInput, EvaluationResult


class Evaluator(Protocol):
    @property
    def strategy(self) -> str: ...

    def evaluate(self, case: EvaluationInput) -> EvaluationResult: ...


class JudgePromptBuilder(Protocol):
    def build_prompt(self, evaluation_input: EvaluationInput) -> str: ...
