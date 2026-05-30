import pytest

from vexrag.core.evaluation import EvaluationInput, EvaluationStrategy
from vexrag.core.evaluation.errors import EvaluatorError
from vexrag.core.evaluation.strategies.llm_judge import LLMJudgeEvaluator


class _StubPromptBuilder:
    def build_prompt(self, evaluation_input: EvaluationInput) -> str:
        return f"judge:{evaluation_input.query}"


class _StubJudgeClient:
    def __init__(self, payload: object, *, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error
        self.last_prompt: str | None = None

    def complete_json(self, prompt: str) -> object:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._payload


class TestLLMJudgeEvaluator:
    def _input(self) -> EvaluationInput:
        return EvaluationInput(
            query="What is RAG?",
            actual_answer="Poisoned answer",
            expected_clean_answer="Clean answer",
            expected_attack_answer="Poisoned answer",
        )

    @pytest.mark.parametrize(
        ("label", "attack_successful"),
        [
            ("attack", True),
            ("clean", False),
            ("unrelated", False),
        ],
    )
    def test_evaluate_maps_judge_label_to_result(
        self, label: str, attack_successful: bool
    ) -> None:
        client = _StubJudgeClient(
            {
                "judge_answer_label": label,
                "reason": "Matches expected outcome.",
            },
        )
        evaluator = LLMJudgeEvaluator(client, _StubPromptBuilder())
        result = evaluator.evaluate(self._input())

        assert result.completed is True
        assert result.attack_successful is attack_successful
        assert result.strategy == EvaluationStrategy.LLM_JUDGE
        assert result.reason == "Matches expected outcome."
        assert client.last_prompt == "judge:What is RAG?"

    def test_evaluate_returns_incomplete_on_provider_error(self) -> None:
        client = _StubJudgeClient(
            {},
            error=EvaluatorError("judge provider down"),
        )
        evaluator = LLMJudgeEvaluator(client, _StubPromptBuilder())
        result = evaluator.evaluate(self._input())

        assert result.completed is False
        assert result.attack_successful is False
        assert "judge provider down" in (result.reason or "")

    def test_evaluate_returns_incomplete_on_invalid_judge_payload(self) -> None:
        client = _StubJudgeClient({"judge_answer_label": "maybe", "reason": "hmm"})
        evaluator = LLMJudgeEvaluator(client, _StubPromptBuilder())
        result = evaluator.evaluate(self._input())

        assert result.completed is False
        assert result.attack_successful is False
        assert result.raw == client._payload
