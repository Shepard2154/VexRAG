from typing import Any

from vexrag.core.adversarial_probe import probe_with_poisoning_and_evaluation
from vexrag.core.evaluation import EvaluationInput, EvaluationResult
from vexrag.core.target import TargetSystemQuery, TargetSystemResponse


class _FakeEval:
    def __init__(self) -> None:
        self.last_input: EvaluationInput | None = None

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        self.last_input = evaluation_input
        return EvaluationResult(
            attack_successful=False,
            strategy="stub",
            scores={},
        )


class _FakeTarget:
    def __init__(self) -> None:
        self.last_query: TargetSystemQuery | None = None

    def answer(self, request: TargetSystemQuery) -> TargetSystemResponse:
        self.last_query = request
        return TargetSystemResponse(answer="stub answer", contexts=())


def test_probe_runs_target_and_evaluation_without_corpus() -> None:
    target = _FakeTarget()
    evaluation = _FakeEval()
    metadata: dict[str, Any] = {"probe": True}

    response, result = probe_with_poisoning_and_evaluation(
        query="q?",
        correct_answer="clean",
        incorrect_answer="bad",
        adversarial_texts=("ctx1",),
        corpus_poisoner=None,
        target_system=target,
        evaluation_strategy=evaluation,
        override_contexts=False,
        cleanup=False,
        metadata=metadata,
    )

    assert response.answer == "stub answer"
    assert result.strategy == "stub"
    assert target.last_query is not None
    assert target.last_query.query == "q?"
    assert evaluation.last_input is not None
    assert evaluation.last_input.expected_clean_answer == "clean"
