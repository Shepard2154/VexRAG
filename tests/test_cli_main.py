from dataclasses import dataclass, field
from pathlib import Path

import pytest

from vexrag.cli import main as cli_main
from vexrag.cli.handlers import doctor as doctor_handler
from vexrag.cli.handlers import generate_cases as generate_cases_handler
from vexrag.cli.handlers import scan as scan_handler
from vexrag.usecases import doctor as doctor_usecase
from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.types import GenerateCasesResult


@dataclass(frozen=True, slots=True)
class _DummyEvaluation:
    attack_successful: bool
    completed: bool
    strategy: str = "stub"
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    raw: object | None = None
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
        return self.evaluation.completed and self.evaluation.attack_successful


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
        return sum(1 for case in self.cases if case.evaluation.completed)

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
                completed=True,
            ),
            case_id="case-1",
        )
        if on_case_complete is not None:
            on_case_complete(case)
        return _DummyReport(
            verdict=_DummyVerdict("not_vulnerable"),
            cases=(case,),
        )


def _stub_scan_dependencies(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(scan_handler, "load_config", lambda _: {})
    monkeypatch.setattr(scan_handler, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(scan_handler, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(scan_handler, "preflight_vllm_models", lambda _: None)
    monkeypatch.setattr(
        scan_handler,
        "build_scan_command",
        lambda *_args, **_kwargs: _DummyCommand(),
    )
    monkeypatch.setattr(
        scan_handler,
        "run_scan",
        lambda command, *, on_case_complete=None: command.run(
            on_case_complete=on_case_complete
        ),
    )


def test_doctor_command_passes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor_handler, "load_config", lambda _: {})
    monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", lambda _: None)
    monkeypatch.setattr(
        doctor_usecase,
        "build_scan_command",
        lambda *_args, **_kwargs: _DummyCommand(),
    )
    status = cli_main.main(["doctor", "--config", str(Path("scan.yml"))])
    captured = capsys.readouterr()
    assert status == 0
    assert "VexRAG Doctor" in captured.out
    assert "Doctor verdict: PASS" in captured.out


def test_doctor_command_fails_when_check_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor_handler, "load_config", lambda _: {})

    def _fail_target(_config: object) -> None:
        raise RuntimeError("target down")

    monkeypatch.setattr(doctor_usecase, "preflight_target_system", _fail_target)
    monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", lambda _: None)
    monkeypatch.setattr(
        doctor_usecase,
        "build_scan_command",
        lambda *_args, **_kwargs: _DummyCommand(),
    )
    status = cli_main.main(["doctor", "--config", str(Path("scan.yml"))])
    captured = capsys.readouterr()
    assert status == 1
    assert "[FAIL] target API availability" in captured.out
    assert "Doctor verdict: FAIL" in captured.out


def test_doctor_usecase_passes_without_vllm_sections(monkeypatch) -> None:
    monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(
        doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
    )
    result = doctor_usecase.run_doctor(config={}, base_dir=Path("."), check_llms=False)
    assert all(check.ok for check in result.checks)
    assert any(
        check.name == "vLLM endpoint + required models" for check in result.checks
    )


def test_doctor_usecase_fails_when_vllm_endpoint_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(
        doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
    )

    def _fail_vllm(_config: object) -> None:
        raise UseCaseDependencyError("could not query vLLM model list")

    monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", _fail_vllm)
    result = doctor_usecase.run_doctor(config={}, base_dir=Path("."), check_llms=False)
    vllm_check = next(
        check
        for check in result.checks
        if check.name == "vLLM endpoint + required models"
    )
    assert vllm_check.ok is False
    assert "could not query vLLM model list" in (vllm_check.error or "")


def test_doctor_usecase_fails_when_vllm_model_missing(monkeypatch) -> None:
    monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(
        doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
    )

    def _fail_vllm(_config: object) -> None:
        raise UseCaseDependencyError("vLLM model(s) not available")

    monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", _fail_vllm)
    result = doctor_usecase.run_doctor(config={}, base_dir=Path("."), check_llms=False)
    vllm_check = next(
        check
        for check in result.checks
        if check.name == "vLLM endpoint + required models"
    )
    assert vllm_check.ok is False
    assert "vLLM model(s) not available" in (vllm_check.error or "")


def test_generate_cases_command_prints_summary(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    output_path = tmp_path / "cases.yml"

    def _stub_run(*_args, **_kwargs) -> GenerateCasesResult:
        return GenerateCasesResult(
            attack_id="hijackrag",
            display_name="HijackRAG",
            output_path=output_path,
            case_count=3,
            topic="widgets",
            adv_per_query=2,
        )

    monkeypatch.setattr(generate_cases_handler, "load_config", lambda _: {})
    monkeypatch.setattr(generate_cases_handler, "run_generate_cases", _stub_run)
    status = cli_main.main(
        [
            "generate-cases",
            "--config",
            str(Path("scan.yml")),
            "--attack",
            "hijackrag",
            "--output",
            str(output_path),
            "--count",
            "3",
            "--topic",
            "widgets",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert "Generated HijackRAG cases" in captured.out
    assert str(output_path) in captured.out
    assert "Cases: 3" in captured.out
    assert "Topic: widgets" in captured.out
    assert "case_files: ['" in captured.out
    assert "adv_per_query: 2" in captured.out


def test_generate_cases_quiet_prints_path_only(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    output_path = tmp_path / "cases.yml"

    def _stub_run(*_args, **_kwargs) -> GenerateCasesResult:
        return GenerateCasesResult(
            attack_id="hijackrag",
            display_name="HijackRAG",
            output_path=output_path,
            case_count=1,
            topic=None,
            adv_per_query=1,
        )

    monkeypatch.setattr(generate_cases_handler, "load_config", lambda _: {})
    monkeypatch.setattr(generate_cases_handler, "run_generate_cases", _stub_run)
    status = cli_main.main(
        [
            "generate-cases",
            "--config",
            str(Path("scan.yml")),
            "--attack",
            "hijackrag",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert captured.out.strip() == str(output_path)


def test_usecases_have_no_print_calls() -> None:
    usecases_dir = Path(__file__).resolve().parents[1] / "vexrag" / "usecases"
    offenders: list[str] = []
    for path in sorted(usecases_dir.rglob("*.py")):
        if "print(" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(usecases_dir.parent.parent)))
    assert offenders == []


def test_scan_uses_compact_output_by_default(monkeypatch, capsys) -> None:
    _stub_scan_dependencies(monkeypatch)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
    captured = capsys.readouterr()
    assert status == 0
    assert "Case details:" not in captured.out
    assert "Final verdict:" in captured.out
    assert captured.out.count("Final verdict:") == 1
    assert captured.out.count("ASR:") == 1


def test_scan_detailed_prints_case_details(monkeypatch, capsys) -> None:
    _stub_scan_dependencies(monkeypatch)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml")), "--detailed"])
    captured = capsys.readouterr()
    assert status == 0
    assert "Case details:" in captured.out


def test_scan_quiet_suppresses_detailed_even_with_detailed_flag(
    monkeypatch, capsys
) -> None:
    _stub_scan_dependencies(monkeypatch)
    status = cli_main.main(
        [
            "scan",
            "--config",
            str(Path("scan.yml")),
            "--quiet",
            "--detailed",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert "Case details:" not in captured.out
    assert "VexRAG Scan" not in captured.out


def test_scan_quiet_still_prints_warnings(monkeypatch, capsys) -> None:
    _stub_scan_dependencies(monkeypatch)

    class _WarningCommand(_DummyCommand):
        def run(self, *, on_case_complete=None):  # type: ignore[no-untyped-def]
            report = super().run(on_case_complete=on_case_complete)
            return _DummyReport(
                verdict=report.verdict,
                cases=report.cases,
                warnings=("quiet-mode warning",),
            )

    monkeypatch.setattr(
        scan_handler,
        "build_scan_command",
        lambda *_args, **_kwargs: _WarningCommand(),
    )
    status = cli_main.main(["scan", "--config", str(Path("scan.yml")), "--quiet"])
    captured = capsys.readouterr()
    assert status == 0
    assert "Warnings:" in captured.out
    assert "quiet-mode warning" in captured.out


def test_scan_build_passes_probe_llms(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    def _capture_build(*_args, **kwargs) -> _DummyCommand:
        captured["probe_llms"] = bool(kwargs.get("probe_llms"))
        return _DummyCommand()

    _stub_scan_dependencies(monkeypatch)
    monkeypatch.setattr(scan_handler, "build_scan_command", _capture_build)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
    assert status == 0
    assert captured.get("probe_llms") is True


def test_scan_calls_vllm_preflight(monkeypatch) -> None:
    called: dict[str, bool] = {"vllm": False}

    def _capture_vllm(_config: object) -> None:
        called["vllm"] = True

    _stub_scan_dependencies(monkeypatch)
    monkeypatch.setattr(scan_handler, "preflight_vllm_models", _capture_vllm)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
    assert status == 0
    assert called["vllm"] is True


def test_cli_package_imports() -> None:
    import vexrag.cli  # noqa: F401


def test_config_error_exit_code(monkeypatch) -> None:
    def _raise_config(_path: Path) -> dict:
        raise UseCaseConfigError("bad config")

    monkeypatch.setattr(scan_handler, "load_config", _raise_config)
    status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
    assert status == 2


def test_version_flag_prints_and_exits_zero(monkeypatch, capsys) -> None:
    from vexrag.cli import parser as cli_parser

    monkeypatch.setattr(cli_parser, "distribution_version", lambda: "0.0.0-test")
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "VexRAG 0.0.0-test" in out


def test_short_version_flag_prints_and_exits_zero(monkeypatch, capsys) -> None:
    from vexrag.cli import parser as cli_parser

    monkeypatch.setattr(cli_parser, "distribution_version", lambda: "0.0.0-test")
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["-V"])
    assert exc_info.value.code == 0
    assert "VexRAG 0.0.0-test" in capsys.readouterr().out
