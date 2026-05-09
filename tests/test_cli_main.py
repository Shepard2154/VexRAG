from dataclasses import dataclass, field
from pathlib import Path

import pytest

from vexrag.cli import main as cli_main


@dataclass(frozen=True, slots=True)
class _DummyEvaluation:
    attack_successful: bool
    evaluation_completed: bool
    strategy: str = "stub"
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    raw_response: object | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DummySystemResponse:
    answer: str
    contexts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DummyCase:
    query: str
    adversarial_texts: tuple[str, ...]
    expected_incorrect_answer: str
    system_response: _DummySystemResponse
    evaluation: _DummyEvaluation
    case_id: str | None = None
    run_index: int = 1
    warnings: tuple[str, ...] = ()

    @property
    def successful(self) -> bool:
        return self.evaluation.evaluation_completed and self.evaluation.attack_successful


@dataclass(frozen=True, slots=True)
class _DummyVerdict:
    value: str


@dataclass(frozen=True, slots=True)
class _DummyReport:
    verdict: _DummyVerdict
    cases: tuple[_DummyCase, ...]
    warnings: tuple[str, ...] = ()

    @property
    def successful_cases(self) -> int:
        return sum(case.successful for case in self.cases)

    @property
    def evaluated_cases(self) -> int:
        return sum(1 for case in self.cases if case.evaluation.evaluation_completed)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def success_rate(self) -> float:
        if self.evaluated_cases == 0:
            return 0.0
        return self.successful_cases / self.evaluated_cases


class _DummyCommand:
    requests: tuple[object, ...] = ()

    def run(self, *, on_case_complete=None):  # type: ignore[no-untyped-def]
        case = _DummyCase(
            query="query",
            adversarial_texts=("adv",),
            expected_incorrect_answer="wrong",
            system_response=_DummySystemResponse(answer="answer", contexts=("ctx",)),
            evaluation=_DummyEvaluation(
                attack_successful=True,
                evaluation_completed=True,
            ),
            case_id="case-1",
        )
        if on_case_complete is not None:
            on_case_complete(case)
        return _DummyReport(
            verdict=_DummyVerdict("not_vulnerable"),
            cases=(case,),
        )


def _stub_cli_dependencies(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli_main, "_load_config", lambda _: {})
    monkeypatch.setattr(cli_main, "_log_config_summary", lambda _: None)
    monkeypatch.setattr(cli_main, "_run_preflight_checks", lambda _: None)
    monkeypatch.setattr(
        cli_main,
        "build_scan_command",
        lambda *_args, **_kwargs: _DummyCommand(),
    )


def test_doctor_command_passes(monkeypatch, capsys) -> None:
    _stub_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli_main, "_preflight_target_system", lambda _: None)
    monkeypatch.setattr(cli_main, "_preflight_ollama_models", lambda _: None)
    status = cli_main.main(["doctor", "--config", str(Path("scan.yml"))])
    captured = capsys.readouterr()
    assert status == 0
    assert "VexRAG Doctor" in captured.out
    assert "Doctor verdict: PASS" in captured.out


def test_scan_uses_compact_output_by_default(monkeypatch, capsys) -> None:
    _stub_cli_dependencies(monkeypatch)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
    captured = capsys.readouterr()
    assert status == 0
    assert "Case details:" not in captured.out
    assert "Final verdict:" in captured.out


def test_scan_detailed_prints_case_details(monkeypatch, capsys) -> None:
    _stub_cli_dependencies(monkeypatch)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml")), "--detailed"])
    captured = capsys.readouterr()
    assert status == 0
    assert "Case details:" in captured.out


def test_version_flag_prints_and_exits_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main, "_distribution_version", lambda: "0.0.0-test")
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "VexRAG 0.0.0-test" in out


def test_short_version_flag_prints_and_exits_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main, "_distribution_version", lambda: "0.0.0-test")
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["-V"])
    assert exc_info.value.code == 0
    assert "VexRAG 0.0.0-test" in capsys.readouterr().out
