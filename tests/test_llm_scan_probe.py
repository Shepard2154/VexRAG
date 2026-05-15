import pytest

from vexrag.core.llm_scan_probe import _probe_complete_json
from vexrag.core.providers.errors import ProviderServiceError


class _FailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise ProviderServiceError("simulated LLM failure")


def test_probe_complete_json_wraps_provider_service_error() -> None:
    with pytest.raises(ProviderServiceError, match="LLM unavailable for scan"):
        _probe_complete_json(_FailingLLM(), role="unit test")


class _UnexpectedFailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM failure")


def test_probe_complete_json_propagates_unexpected_errors() -> None:
    with pytest.raises(RuntimeError, match="simulated LLM failure"):
        _probe_complete_json(_UnexpectedFailingLLM(), role="unit test")
