from collections.abc import Callable
from typing import Any, Protocol

from vexrag.core.evaluation import EvaluationResult
from vexrag.core.target import TargetSystemResponse


class ScanCaseReportProtocol(Protocol):
    query: str
    adversarial_texts: tuple[str, ...]
    expected_incorrect_answer: str
    system_response: TargetSystemResponse
    evaluation: EvaluationResult
    case_id: str | None
    run_index: int

    @property
    def successful(self) -> bool: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...


class ScanVerdictProtocol(Protocol):
    value: str


class ScanReportProtocol(Protocol):
    verdict: ScanVerdictProtocol
    cases: tuple[ScanCaseReportProtocol, ...]
    warnings: tuple[str, ...]

    @property
    def success_rate(self) -> float: ...

    @property
    def successful_cases(self) -> int: ...

    @property
    def total_cases(self) -> int: ...


class ScanCommandProtocol(Protocol):
    """Runnable scan assembled from config (CLI, web, or programmatic)."""

    def run(
        self,
        *,
        on_case_complete: Callable[[ScanCaseReportProtocol], None] | None = None,
    ) -> ScanReportProtocol: ...


class ConfiguredScanCommand:
    """Wires a runner with requests and scan config."""
    __slots__ = ("runner", "requests", "scan_config")

    def __init__(self, runner: Any, requests: Any, scan_config: Any) -> None:
        self.runner = runner
        self.requests = requests
        self.scan_config = scan_config

    def run(
        self,
        *,
        on_case_complete: Callable[[ScanCaseReportProtocol], None] | None = None,
    ) -> ScanReportProtocol:
        return self.runner.run(
            self.requests,
            self.scan_config,
            on_case_complete=on_case_complete,
        )
