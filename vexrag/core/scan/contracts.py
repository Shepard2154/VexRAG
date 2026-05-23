from collections.abc import Callable
from typing import Protocol

from vexrag.core.evaluation.types import EvaluationResult
from vexrag.core.target_systems.types import TargetSystemResponse


class ScanCaseReport(Protocol):
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


class ScanVerdictValue(Protocol):
    value: str


class ScanReport(Protocol):
    verdict: ScanVerdictValue
    cases: tuple[ScanCaseReport, ...]
    warnings: tuple[str, ...]

    @property
    def success_rate(self) -> float: ...

    @property
    def successful_cases(self) -> int: ...

    @property
    def evaluated_cases(self) -> int: ...

    @property
    def total_cases(self) -> int: ...


class ScanCommand(Protocol):
    """Runnable scan assembled from config (CLI, web, or programmatic)."""

    def run(
        self,
        *,
        on_case_complete: Callable[[ScanCaseReport], None] | None = None,
    ) -> ScanReport: ...
