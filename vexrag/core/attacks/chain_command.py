from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vexrag.core.attacks.command import (
    ScanCaseReportProtocol,
    ScanCommandProtocol,
    ScanReportProtocol,
)
from vexrag.core.contracts.scan import ScanVerdict


class _ChainLabeledCase:
    """Wraps a case report so the printed ``case_id`` includes the chain step."""

    __slots__ = ("_inner", "_label")

    def __init__(
        self,
        inner: ScanCaseReportProtocol,
        *,
        step_index: int,
        attack_id: str,
    ) -> None:
        self._inner = inner
        self._label = f"[step {step_index} {attack_id}]"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def case_id(self) -> str | None:
        base = self._inner.case_id
        tail = base.strip() if isinstance(base, str) else ""
        combined = f"{self._label} {tail}".strip()
        return combined or self._label


@dataclass(frozen=True, slots=True)
class AttackChainScanReport:
    """Aggregated report after running every step in an attack chain."""

    step_reports: tuple[tuple[str, ScanReportProtocol], ...]
    verdict: ScanVerdict
    cases: tuple[ScanCaseReportProtocol, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def success_rate(self) -> float:
        evaluated = self.evaluated_cases
        if evaluated == 0:
            return 0.0
        hits = sum(
            1
            for case in self.cases
            if case.evaluation.completed and case.evaluation.attack_successful
        )
        return hits / evaluated

    @property
    def successful_cases(self) -> int:
        return sum(case.successful for case in self.cases)

    @property
    def evaluated_cases(self) -> int:
        return sum(1 for case in self.cases if case.evaluation.completed)

    @property
    def total_cases(self) -> int:
        return len(self.cases)


class AttackChainScanCommand:
    """Runs each subcommand in order and merges case reports."""

    __slots__ = ("_steps",)

    def __init__(self, steps: tuple[tuple[str, ScanCommandProtocol], ...]) -> None:
        self._steps = steps

    @property
    def requests(self) -> tuple[Any, ...]:
        merged: list[Any] = []
        for _, command in self._steps:
            chunk = getattr(command, "requests", ())
            if chunk:
                merged.extend(chunk)
        return tuple(merged)

    def run(
        self,
        *,
        on_case_complete: Callable[[ScanCaseReportProtocol], None] | None = None,
    ) -> AttackChainScanReport:
        step_reports: list[tuple[str, ScanReportProtocol]] = []
        flat_cases: list[ScanCaseReportProtocol] = []
        warn: list[str] = []
        any_vulnerable = False
        step_index = 0
        for attack_id, command in self._steps:
            step_index += 1
            sid = step_index
            aid = attack_id

            def step_callback(
                case: ScanCaseReportProtocol,
                *,
                _sid: int = sid,
                _aid: str = aid,
            ) -> None:
                if on_case_complete is None:
                    return
                labeled = _ChainLabeledCase(
                    case,
                    step_index=_sid,
                    attack_id=_aid,
                )
                on_case_complete(labeled)

            report = command.run(
                on_case_complete=(
                    step_callback if on_case_complete is not None else None
                ),
            )
            step_reports.append((attack_id, report))
            warn.extend(report.warnings)
            if report.verdict is ScanVerdict.VULNERABLE:
                any_vulnerable = True
            for case in report.cases:
                flat_cases.append(
                    _ChainLabeledCase(
                        case,
                        step_index=step_index,
                        attack_id=attack_id,
                    )
                )
        verdict = (
            ScanVerdict.VULNERABLE if any_vulnerable else ScanVerdict.NOT_VULNERABLE
        )
        return AttackChainScanReport(
            step_reports=tuple(step_reports),
            verdict=verdict,
            cases=tuple(flat_cases),
            warnings=tuple(dict.fromkeys(warn)),
        )


def format_chain_step_summary_lines(
    step_reports: Sequence[tuple[str, ScanReportProtocol]],
) -> tuple[str, ...]:
    lines: list[str] = []
    for step_index, (attack_id, report) in enumerate(step_reports, start=1):
        lines.append(
            f"  Step {step_index} ({attack_id}): verdict={report.verdict.value.upper()} "
            f"ASR={report.success_rate:.2%} ({report.successful_cases}/"
            f"{report.evaluated_cases} evaluated)"
        )
    return tuple(lines)
