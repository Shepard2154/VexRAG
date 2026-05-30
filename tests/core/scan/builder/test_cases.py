import pytest

from vexrag.core.scan.builder.cases import (
    case_configs_from_value,
    load_case_configs,
    path_strings_from_value,
)
from vexrag.core.scan.config.errors import ScanConfigError


class TestPathStringsFromValue:
    def test_path_strings_accepts_single_string(self) -> None:
        assert path_strings_from_value("./cases/a.yaml", "case_files") == (
            "./cases/a.yaml",
        )

    def test_path_strings_accepts_list(self) -> None:
        assert path_strings_from_value(["a.yaml", "b.yaml"], "case_files") == (
            "a.yaml",
            "b.yaml",
        )

    def test_path_strings_rejects_invalid_type(self) -> None:
        with pytest.raises(ScanConfigError, match="must be a string or a list"):
            path_strings_from_value(42, "case_files")


class TestCaseConfigsFromValue:
    def test_case_configs_from_inline_list(self) -> None:
        cases = case_configs_from_value(
            [{"query": "q1"}, {"query": "q2"}],
            "cases",
        )
        assert len(cases) == 2
        assert cases[0]["query"] == "q1"

    def test_case_configs_rejects_non_mapping_entry(self) -> None:
        with pytest.raises(ScanConfigError, match=r"cases\[2\] must be a mapping"):
            case_configs_from_value([{"query": "ok"}, "bad"], "cases")


class TestLoadCaseConfigs:
    def test_load_yaml_cases_file(self, tmp_path) -> None:
        cases_file = tmp_path / "cases.yaml"
        cases_file.write_text(
            "cases:\n  - id: one\n    query: What is RAG?\n    correct_answer: ok\n",
            encoding="utf-8",
        )
        loaded = load_case_configs("cases.yaml", base_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "one"
        assert loaded[0]["query"] == "What is RAG?"

    def test_load_jsonl_cases_file(self, tmp_path) -> None:
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text(
            '{"query": "q1", "correct_answer": "a1"}\n'
            '{"query": "q2", "correct_answer": "a2"}\n',
            encoding="utf-8",
        )
        loaded = load_case_configs(str(cases_file), base_dir=None)
        assert len(loaded) == 2
        assert loaded[1]["query"] == "q2"

    def test_load_cases_resolves_relative_path_with_base_dir(self, tmp_path) -> None:
        sub = tmp_path / "data"
        sub.mkdir()
        (sub / "cases.yaml").write_text(
            "cases:\n  - query: relative\n",
            encoding="utf-8",
        )
        loaded = load_case_configs("data/cases.yaml", base_dir=tmp_path)
        assert loaded[0]["query"] == "relative"

    def test_load_cases_rejects_invalid_yaml(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("cases: [unclosed", encoding="utf-8")
        with pytest.raises(ScanConfigError, match="could not parse cases file"):
            load_case_configs(str(bad), base_dir=None)

    def test_load_cases_rejects_non_list_root(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("cases: not-a-list\n", encoding="utf-8")
        with pytest.raises(ScanConfigError, match="must be a list of case mappings"):
            load_case_configs(str(bad), base_dir=None)
