import pytest

from vexrag.core.errors import ProviderServiceError
from vexrag.core.llm_scan_probe import _probe_complete_json


class _FailingLLM:
    def complete_json(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM failure")


def test_probe_complete_json_wraps_underlying_error() -> None:
    with pytest.raises(ProviderServiceError, match="LLM unavailable for scan"):
        _probe_complete_json(_FailingLLM(), role="unit test")
