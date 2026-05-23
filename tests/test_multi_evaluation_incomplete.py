from vexrag.core.evaluation import (
    CombineMode,
    CompositeEvaluator,
    EvaluationInput,
    EvaluationResult,
)


class _OkEvaluator:
    strategy = "ok"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=True,
            strategy=self.strategy,
        )


class _BrokenEvaluator:
    strategy = "broken"

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=False,
            completed=False,
            strategy=self.strategy,
            reason="embedding failed",
        )


def test_composite_any_succeeds_when_one_sub_evaluator_succeeds() -> None:
    composite = CompositeEvaluator(
        (_OkEvaluator(), _BrokenEvaluator()),
        combine=CombineMode.ANY,
    )
    r = composite.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.completed is True
    assert r.attack_successful is True
    assert r.children is not None
    assert len(r.children) == 2


def test_composite_all_marks_incomplete_when_sub_evaluator_fails() -> None:
    composite = CompositeEvaluator(
        (_OkEvaluator(), _BrokenEvaluator()),
        combine=CombineMode.ALL,
    )
    r = composite.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.completed is False
    assert r.attack_successful is False
