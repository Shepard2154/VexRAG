from collections.abc import Callable
from typing import Protocol

from vexrag.core.evaluation.types import EvaluationResult
from vexrag.core.target_systems.types import TargetSystemResponse


class ScanCaseReport(Protocol):
    @property
    def query(self) -> str: ...

    @property
    def adversarial_texts(self) -> tuple[str, ...]: ...

    @property
    def expected_incorrect_answer(self) -> str: ...

    @property
    def system_response(self) -> TargetSystemResponse: ...

    @property
    def evaluation(self) -> EvaluationResult: ...

    @property
    def case_id(self) -> str | None: ...

    @property
    def run_index(self) -> int: ...

    @property
    def successful(self) -> bool: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...


class ScanVerdictValue(Protocol):
    value: str


class ScanReport(Protocol):
    @property
    def verdict(self) -> ScanVerdictValue: ...

    @property
    def cases(self) -> tuple[ScanCaseReport, ...]: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...

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
