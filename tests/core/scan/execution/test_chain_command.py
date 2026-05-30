from vexrag.attack_algorithms.poison_base.report import (
    CorpusPoisonCaseResult,
    CorpusPoisonScanReport,
)
from vexrag.core.evaluation import EvaluationResult, EvaluationStrategy
from vexrag.core.scan.execution.chain_command import AttackChainScanCommand
from vexrag.core.scan.types import ScanVerdict
from vexrag.core.target_systems import TargetSystemResponse


def _case(
    *,
    case_id: str,
    attack_successful: bool,
    completed: bool = True,
) -> CorpusPoisonCaseResult:
    return CorpusPoisonCaseResult(
        query="q",
        adversarial_texts=("adv",),
        expected_incorrect_answer="wrong",
        system_response=TargetSystemResponse(answer="a", contexts=()),
        evaluation=EvaluationResult(
            attack_successful=attack_successful,
            completed=completed,
            strategy=EvaluationStrategy.LLM_JUDGE,
        ),
        case_id=case_id,
    )


def _report(
    *,
    verdict: ScanVerdict,
    cases: tuple[CorpusPoisonCaseResult, ...],
) -> CorpusPoisonScanReport:
    return CorpusPoisonScanReport(
        verdict=verdict,
        cases=cases,
        generated_adversarial_texts=(),
    )


class _StubScanCommand:
    def __init__(
        self, report: CorpusPoisonScanReport, requests: tuple[object, ...]
    ) -> None:
        self._report = report
        self.requests = requests

    def run(self, *, on_case_complete=None):  # type: ignore[no-untyped-def]
        if on_case_complete is not None:
            for case in self._report.cases:
                on_case_complete(case)
        return self._report


class TestAttackChainScanCommand:
    def test_merged_report_labels_cases_with_step_prefix(self) -> None:
        step1 = _StubScanCommand(
            _report(
                verdict=ScanVerdict.NOT_VULNERABLE,
                cases=(_case(case_id="a", attack_successful=False),),
            ),
            requests=("r1",),
        )
        step2 = _StubScanCommand(
            _report(
                verdict=ScanVerdict.VULNERABLE,
                cases=(_case(case_id="b", attack_successful=True),),
            ),
            requests=("r2", "r3"),
        )
        chain = AttackChainScanCommand(
            (("hijackrag", step1), ("poisonedrag", step2)),
        )

        merged = chain.run()
        assert merged.verdict.value == ScanVerdict.VULNERABLE
        assert len(merged.cases) == 2
        assert merged.cases[0].case_id == "[step 1 hijackrag] a"
        assert merged.cases[1].case_id == "[step 2 poisonedrag] b"
        assert chain.requests == ("r1", "r2", "r3")

    def test_merged_success_rate_counts_evaluated_cases(self) -> None:
        step1 = _StubScanCommand(
            _report(
                verdict=ScanVerdict.NOT_VULNERABLE,
                cases=(
                    _case(case_id="a", attack_successful=True),
                    _case(case_id="b", attack_successful=False),
                ),
            ),
            requests=(),
        )
        step2 = _StubScanCommand(
            _report(
                verdict=ScanVerdict.NOT_VULNERABLE,
                cases=(_case(case_id="c", attack_successful=True, completed=False),),
            ),
            requests=(),
        )
        merged = AttackChainScanCommand(
            (("hijackrag", step1), ("poisonedrag", step2))
        ).run()

        assert merged.evaluated_cases == 2
        assert merged.successful_cases == 1
        assert merged.success_rate == 0.5
        assert merged.verdict.value == ScanVerdict.NOT_VULNERABLE
