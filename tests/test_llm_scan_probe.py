import pytest

from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.execution import probe_complete_json


class _FailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise ProviderServiceError("simulated LLM failure")


def test_probe_complete_json_wraps_provider_service_error() -> None:
    with pytest.raises(ProviderServiceError, match="LLM unavailable for scan"):
        probe_complete_json(_FailingLLM(), role="unit test")


class _UnexpectedFailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM failure")


def test_probe_complete_json_propagates_unexpected_errors() -> None:
    with pytest.raises(RuntimeError, match="simulated LLM failure"):
        probe_complete_json(_UnexpectedFailingLLM(), role="unit test")
