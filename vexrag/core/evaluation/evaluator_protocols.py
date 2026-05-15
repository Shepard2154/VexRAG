from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from vexrag.core.evaluation.attack_verdict import EvaluationResult, EvaluationStrategy
from vexrag.core.evaluation.scan_case_input import EvaluationInput


class Evaluator(Protocol):
    @property
    def strategy(self) -> EvaluationStrategy | str: ...

    def evaluate(self, case: EvaluationInput) -> EvaluationResult: ...


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class JudgeClient(Protocol):
    def complete_json(self, prompt: str) -> str | Mapping[str, Any]: ...


class JudgePromptBuilder(Protocol):
    def build_prompt(self, evaluation_input: EvaluationInput) -> str: ...
