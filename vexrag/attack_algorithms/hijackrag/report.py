from dataclasses import dataclass, field

from vexrag.core.evaluation import EvaluationResult
from vexrag.core.scan.types import ScanVerdict
from vexrag.core.target_systems import TargetSystemResponse


@dataclass(frozen=True, slots=True)
class HijackRAGCaseResult:
    """Single HijackRAG target-system check."""

    query: str
    adversarial_texts: tuple[str, ...]
    expected_incorrect_answer: str
    system_response: TargetSystemResponse
    evaluation: EvaluationResult
    case_id: str | None = None
    run_index: int = 1

    @property
    def successful(self) -> bool:
        return self.evaluation.completed and self.evaluation.attack_successful

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.evaluation.warnings


@dataclass(frozen=True, slots=True)
class HijackRAGScanReport:
    """Machine-readable report for a HijackRAG scan."""

    verdict: ScanVerdict
    cases: tuple[HijackRAGCaseResult, ...]
    generated_adversarial_texts: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def evaluated_cases(self) -> int:
        return sum(1 for case in self.cases if case.evaluation.completed)

    @property
    def successful_cases(self) -> int:
        return sum(case.successful for case in self.cases)

    @property
    def success_rate(self) -> float:
        evaluated = self.evaluated_cases
        if evaluated == 0:
            return 0.0
        return self.successful_cases / evaluated

    @property
    def vulnerable(self) -> bool:
        return self.verdict is ScanVerdict.VULNERABLE
