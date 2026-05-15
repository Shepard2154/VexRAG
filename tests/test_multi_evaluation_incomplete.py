from vexrag.core.evaluation.multi_evaluator import MultiEvaluator
from vexrag.core.evaluation.protocols import EvaluationInput, EvaluationResult


class _OkEvaluator:
    strategy = "ok"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(True, self.strategy)


class _BrokenEvaluator:
    strategy = "broken"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=False,
            strategy=self.strategy,
            evaluation_completed=False,
        )


def test_multi_yield_incomplete_when_any_sub_evaluator_incomplete() -> None:
    m = MultiEvaluator((_OkEvaluator(), _BrokenEvaluator()), combine="any")
    r = m.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="a",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.evaluation_completed is False
    assert r.attack_successful is False
