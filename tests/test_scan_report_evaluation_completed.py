"""Attack success rate uses only evaluations that completed."""

from vexrag.attack_algorithms.hijackrag.report import (
    HijackRAGCaseResult,
    HijackRAGScanReport,
)
from vexrag.core.evaluation import EvaluationResult
from vexrag.core.scan.types import ScanVerdict
from vexrag.core.target_systems import TargetSystemResponse


def _case(
    *,
    completed: bool,
    attack_successful: bool,
) -> HijackRAGCaseResult:
    return HijackRAGCaseResult(
        query="q",
        adversarial_texts=("adv",),
        expected_incorrect_answer="wrong",
        system_response=TargetSystemResponse(answer="a", contexts=()),
        evaluation=EvaluationResult(
            attack_successful=attack_successful,
            strategy="stub",
            completed=completed,
        ),
    )


def test_hijack_report_success_rate_denominator_counts_evaluated_only() -> None:
    cases = (
        _case(completed=True, attack_successful=True),
        _case(completed=False, attack_successful=False),
    )
    report = HijackRAGScanReport(
        verdict=ScanVerdict.NOT_VULNERABLE,
        cases=cases,
        generated_adversarial_texts=("adv",),
    )
    assert report.total_cases == 2
    assert report.evaluated_cases == 1
    assert report.successful_cases == 1
    assert report.success_rate == 1.0
