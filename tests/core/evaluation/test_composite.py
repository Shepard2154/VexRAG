import pytest

from tests.mocks import FixedEvaluator
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


def _sample_input() -> EvaluationInput:
    return EvaluationInput(
        query="q",
        actual_answer="x",
        expected_clean_answer="c",
        expected_attack_answer="i",
    )


class TestCompositeEvaluator:
    def test_composite_sub_evaluators_exposes_children(self) -> None:
        inner = (FixedEvaluator("a", False), FixedEvaluator("b", True))
        composite = CompositeEvaluator(inner, combine=CombineMode.ANY)
        assert composite.sub_evaluators == inner

    def test_composite_any_true_if_one_succeeds(self) -> None:
        composite = CompositeEvaluator(
            (FixedEvaluator("a", False), FixedEvaluator("b", True)),
            combine=CombineMode.ANY,
        )
        r = composite.evaluate(_sample_input())
        assert r.attack_successful is True
        assert "a" in r.strategy and "b" in r.strategy

    def test_composite_all_false_if_one_fails(self) -> None:
        composite = CompositeEvaluator(
            (FixedEvaluator("a", True), FixedEvaluator("b", False)),
            combine=CombineMode.ALL,
        )
        r = composite.evaluate(_sample_input())
        assert r.attack_successful is False

    def test_composite_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CompositeEvaluator((), combine=CombineMode.ANY)

    def test_composite_any_succeeds_when_one_sub_evaluator_succeeds(self) -> None:
        composite = CompositeEvaluator(
            (_OkEvaluator(), _BrokenEvaluator()),
            combine=CombineMode.ANY,
        )
        r = composite.evaluate(_sample_input())
        assert r.completed is True
        assert r.attack_successful is True
        assert r.children is not None
        assert len(r.children) == 2

    def test_composite_all_marks_incomplete_when_sub_evaluator_fails(self) -> None:
        composite = CompositeEvaluator(
            (_OkEvaluator(), _BrokenEvaluator()),
            combine=CombineMode.ALL,
        )
        r = composite.evaluate(_sample_input())
        assert r.completed is False
        assert r.attack_successful is False
