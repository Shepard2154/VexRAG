from pathlib import Path

import pytest

from vexrag.usecases.config_io import load_config
from vexrag.usecases.errors import UseCaseConfigError
from vexrag.usecases.generate_cases import run_generate_cases


class TestRunGenerateCases:
    def test_run_generate_cases_writes_yaml_and_returns_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "generated.yaml"
        fake_cases = [
            {"id": "c1", "query": "q1"},
            {"id": "c2", "query": "q2"},
        ]

        class _FakePlugin:
            attack_id = "hijackrag"
            display_name = "HijackRAG"

            def generate_cases(self, config, params):  # type: ignore[no-untyped-def]
                return fake_cases

            def serialize_case_for_yaml(self, case):  # type: ignore[no-untyped-def]
                return dict(case)

        class _FakeRegistry:
            def get(self, attack_id: str) -> _FakePlugin:
                assert attack_id == "hijackrag"
                return _FakePlugin()

        monkeypatch.setattr(
            "vexrag.usecases.generate_cases.create_attack_method_registry",
            lambda: _FakeRegistry(),
        )
        monkeypatch.setattr(
            "vexrag.usecases.generate_cases.resolve_generate_cases_attack",
            lambda *_a, **_k: "hijackrag",
        )
        monkeypatch.setattr(
            "vexrag.usecases.generate_cases.materialize_generate_cases_config",
            lambda *_a, **_k: {},
        )

        result = run_generate_cases(
            {},
            attack="hijackrag",
            output=output,
            count=2,
            topic=None,
            target_style="short_fact",
            adv_per_query=1,
            seed=1,
            overwrite=True,
        )

        assert result.case_count == 2
        assert result.output_path == output
        written = load_config(output)
        assert len(written["cases"]) == 2
        assert written["cases"][0]["id"] == "c1"

    def test_run_generate_cases_rejects_existing_output_without_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "existing.yaml"
        output.write_text("cases: []\n", encoding="utf-8")

        monkeypatch.setattr(
            "vexrag.usecases.generate_cases.resolve_generate_cases_attack",
            lambda *_a, **_k: "hijackrag",
        )

        with pytest.raises(UseCaseConfigError, match="already exists"):
            run_generate_cases(
                {},
                attack="hijackrag",
                output=output,
                count=1,
                topic=None,
                target_style="short_fact",
                adv_per_query=1,
                seed=None,
                overwrite=False,
            )
