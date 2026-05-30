from typing import Any

import pytest

from tests.mocks import FakeEval, FakeTarget
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.execution import (
    probe_complete_json,
    probe_with_poisoning_and_evaluation,
)


class TestAdversarialProbe:
    def test_probe_runs_target_and_evaluation_without_corpus(self) -> None:
        target = FakeTarget()
        evaluation = FakeEval()
        metadata: dict[str, Any] = {"probe": True}

        response, result = probe_with_poisoning_and_evaluation(
            query="q?",
            correct_answer="clean",
            incorrect_answer="bad",
            adversarial_texts=("ctx1",),
            corpus_poisoner=None,
            target_system=target,
            evaluator=evaluation,
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


class _FailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise ProviderServiceError("simulated LLM failure")


class _UnexpectedFailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM failure")


class TestLLMScanProbe:
    def test_probe_complete_json_wraps_provider_service_error(self) -> None:
        with pytest.raises(ProviderServiceError, match="LLM unavailable for scan"):
            probe_complete_json(_FailingLLM(), role="unit test")

    def test_probe_complete_json_propagates_unexpected_errors(self) -> None:
        with pytest.raises(RuntimeError, match="simulated LLM failure"):
            probe_complete_json(_UnexpectedFailingLLM(), role="unit test")
