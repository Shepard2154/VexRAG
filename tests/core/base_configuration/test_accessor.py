from pathlib import Path

import pytest

from vexrag.core.base_configuration.accessor import ConfigAccessor
from vexrag.core.exceptions import ConfigurationError


class TestConfigAccessor:
    def test_get_int_rejects_bool(self) -> None:
        accessor = ConfigAccessor({"n": True})
        with pytest.raises(ConfigurationError, match="int"):
            accessor.get_int("n", 0)

    def test_get_required_string_strips(self) -> None:
        accessor = ConfigAccessor({"name": "  x  "})
        assert accessor.get_required_string("name") == "x"

    def test_get_path_resolves_relative_to_base_dir(self, tmp_path: Path) -> None:
        accessor = ConfigAccessor(
            {"cases": "cases.yml"},
            base_dir=tmp_path,
        )
        assert accessor.get_path("cases") == tmp_path / "cases.yml"
