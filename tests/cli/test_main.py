from pathlib import Path

import pytest

from tests.mocks import (
    DummyCommand,
    DummyReport,
    stub_scan_dependencies,
)
from vexrag.cli import main as cli_main
from vexrag.cli.handlers import doctor as doctor_handler
from vexrag.cli.handlers import generate_cases as generate_cases_handler
from vexrag.cli.handlers import scan as scan_handler
from vexrag.usecases import doctor as doctor_usecase
from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.types import GenerateCasesResult

_VLLM_DOCTOR_CONFIG = {
    "evaluation": {
        "provider": "vllm",
        "base_url": "http://127.0.0.1:8000",
        "model": "meta-llama/Llama-3-8B",
    },
}


class TestDoctor:
    def test_doctor_command_passes(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(doctor_handler, "load_config", lambda _: {})
        monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", lambda _: None)
        monkeypatch.setattr(
            doctor_usecase,
            "build_scan_command",
            lambda *_args, **_kwargs: DummyCommand(),
        )
        status = cli_main.main(["doctor", "--config", str(Path("scan.yml"))])
        captured = capsys.readouterr()
        assert status == 0
        assert "VexRAG Doctor" in captured.out
        assert "Doctor verdict: PASS" in captured.out

    def test_doctor_command_fails_when_check_fails(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(doctor_handler, "load_config", lambda _: {})

        def _fail_target(_config: object) -> None:
            raise RuntimeError("target down")

        monkeypatch.setattr(doctor_usecase, "preflight_target_system", _fail_target)
        monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", lambda _: None)
        monkeypatch.setattr(
            doctor_usecase,
            "build_scan_command",
            lambda *_args, **_kwargs: DummyCommand(),
        )
        status = cli_main.main(["doctor", "--config", str(Path("scan.yml"))])
        captured = capsys.readouterr()
        assert status == 1
        assert "[FAIL] target API availability" in captured.out
        assert "Doctor verdict: FAIL" in captured.out

    def test_doctor_usecase_passes_without_vllm_sections(self, monkeypatch) -> None:
        monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
        monkeypatch.setattr(
            doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
        )
        result = doctor_usecase.run_doctor(
            config={}, base_dir=Path("."), check_llms=False
        )
        assert all(check.ok for check in result.checks)
        vllm_check = next(
            check
            for check in result.checks
            if check.name == "vLLM endpoint + required models"
        )
        assert vllm_check.skipped is True

    def test_doctor_usecase_fails_when_vllm_endpoint_unreachable(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
        monkeypatch.setattr(
            doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
        )

        def _fail_vllm(_config: object) -> None:
            raise UseCaseDependencyError("could not query vLLM model list")

        monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", _fail_vllm)
        result = doctor_usecase.run_doctor(
            config=_VLLM_DOCTOR_CONFIG,
            base_dir=Path("."),
            check_llms=False,
        )
        vllm_check = next(
            check
            for check in result.checks
            if check.name == "vLLM endpoint + required models"
        )
        assert vllm_check.skipped is False
        assert vllm_check.ok is False
        assert "could not query vLLM model list" in (vllm_check.error or "")

    def test_doctor_usecase_fails_when_vllm_model_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(doctor_usecase, "preflight_target_system", lambda _: None)
        monkeypatch.setattr(doctor_usecase, "preflight_ollama_models", lambda _: None)
        monkeypatch.setattr(
            doctor_usecase, "build_scan_command", lambda *_args, **_kwargs: None
        )

        def _fail_vllm(_config: object) -> None:
            raise UseCaseDependencyError("vLLM model(s) not available")

        monkeypatch.setattr(doctor_usecase, "preflight_vllm_models", _fail_vllm)
        result = doctor_usecase.run_doctor(
            config=_VLLM_DOCTOR_CONFIG,
            base_dir=Path("."),
            check_llms=False,
        )
        vllm_check = next(
            check
            for check in result.checks
            if check.name == "vLLM endpoint + required models"
        )
        assert vllm_check.skipped is False
        assert vllm_check.ok is False
        assert "vLLM model(s) not available" in (vllm_check.error or "")


class TestGenerateCases:
    def test_generate_cases_command_prints_summary(
        self, monkeypatch, capsys, tmp_path: Path
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
        self, monkeypatch, capsys, tmp_path: Path
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


class TestScanOutput:
    def test_scan_uses_compact_output_by_default(self, monkeypatch, capsys) -> None:
        stub_scan_dependencies(monkeypatch)
        status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
        captured = capsys.readouterr()
        assert status == 0
        assert "Case details:" not in captured.out
        assert "Final verdict:" in captured.out
        assert captured.out.count("Final verdict:") == 1
        assert captured.out.count("ASR:") == 1

    def test_scan_detailed_prints_case_details(self, monkeypatch, capsys) -> None:
        stub_scan_dependencies(monkeypatch)
        status = cli_main.main(
            ["scan", "--config", str(Path("scan.yml")), "--detailed"]
        )
        captured = capsys.readouterr()
        assert status == 0
        assert "Case details:" in captured.out

    def test_scan_quiet_suppresses_detailed_even_with_detailed_flag(
        self, monkeypatch, capsys
    ) -> None:
        stub_scan_dependencies(monkeypatch)
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

    def test_scan_quiet_still_prints_warnings(self, monkeypatch, capsys) -> None:
        stub_scan_dependencies(monkeypatch)

        class _WarningCommand(DummyCommand):
            def run(self, *, on_case_complete=None):  # type: ignore[no-untyped-def]
                report = super().run(on_case_complete=on_case_complete)
                return DummyReport(
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

    def test_scan_build_passes_probe_llms(self, monkeypatch) -> None:
        captured: dict[str, bool] = {}

        def _capture_build(*_args, **kwargs) -> DummyCommand:
            captured["probe_llms"] = bool(kwargs.get("probe_llms"))
            return DummyCommand()

        stub_scan_dependencies(monkeypatch)
        monkeypatch.setattr(scan_handler, "build_scan_command", _capture_build)
        status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
        assert status == 0
        assert captured.get("probe_llms") is True

    def test_scan_calls_vllm_preflight(self, monkeypatch) -> None:
        called: dict[str, bool] = {"vllm": False}

        def _capture_vllm(_config: object) -> None:
            called["vllm"] = True

        stub_scan_dependencies(monkeypatch)
        monkeypatch.setattr(scan_handler, "preflight_vllm_models", _capture_vllm)
        status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
        assert status == 0
        assert called["vllm"] is True

    def test_config_error_exit_code(self, monkeypatch) -> None:
        def _raise_config(_path: Path) -> dict:
            raise UseCaseConfigError("bad config")

        monkeypatch.setattr(scan_handler, "load_config", _raise_config)
        status = cli_main.main(["scan", "--config", str(Path("scan.yml"))])
        assert status == 2


class TestVersion:
    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_version_flag_prints_and_exits_zero(
        self, monkeypatch, capsys, flag: str
    ) -> None:
        from vexrag.cli import parser as cli_parser

        monkeypatch.setattr(cli_parser, "distribution_version", lambda: "0.0.0-test")
        with pytest.raises(SystemExit) as exc_info:
            cli_main.main([flag])
        assert exc_info.value.code == 0
        assert "VexRAG 0.0.0-test" in capsys.readouterr().out
