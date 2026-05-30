from pathlib import Path

import pytest

from vexrag.usecases.config_io import dump_yaml, load_config, load_yaml, write_yaml
from vexrag.usecases.errors import UseCaseConfigError


class TestConfigIO:
    def test_load_yaml_parses_mapping(self) -> None:
        loaded = load_yaml("attack: poisonedrag\n")
        assert loaded == {"attack": "poisonedrag"}

    def test_load_yaml_rejects_invalid_yaml(self) -> None:
        with pytest.raises(UseCaseConfigError, match="invalid"):
            load_yaml("attack: [\n  broken")

    def test_load_config_requires_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.yml"
        path.write_text("- not a mapping\n", encoding="utf-8")
        with pytest.raises(UseCaseConfigError, match="mapping"):
            load_config(path)

    def test_load_config_reads_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.yml"
        path.write_text("top_k: 3\n", encoding="utf-8")
        assert load_config(path)["top_k"] == 3

    def test_dump_and_write_yaml_round_trip(self, tmp_path: Path) -> None:
        content = {"attack": "hijackrag", "nested": {"k": 1}}
        out_path = tmp_path / "out.yml"
        write_yaml(out_path, content)
        assert load_config(out_path) == content
        text = dump_yaml(content)
        assert "attack: hijackrag" in text
