from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from vexrag.core.config.errors import ScanConfigError
from vexrag.core.config_accessor import ConfigAccessor
from vexrag.core.errors import ConfigError

_T = TypeVar("_T")


class ScanConfigAccessor:
    """Adapter that wraps ConfigAccessor and guarantees ScanConfigError."""

    def __init__(
        self,
        config: Mapping[str, Any],
        prefix: str = "",
        base_dir: Path | None = None,
    ) -> None:
        self._inner = ConfigAccessor(config, prefix=prefix, base_dir=base_dir)

    def _with_scan_error(self, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except ConfigError as exc:
            raise ScanConfigError(str(exc)) from exc

    def get_int(self, option_name: str, default_value: int) -> int:
        return self._with_scan_error(
            lambda: self._inner.get_int(option_name, default_value)
        )

    def get_float(self, option_name: str, default_value: float) -> float:
        return self._with_scan_error(
            lambda: self._inner.get_float(option_name, default_value)
        )

    def get_optional_int(
        self, option_name: str, default_value: int | None = None
    ) -> int | None:
        return self._with_scan_error(
            lambda: self._inner.get_optional_int(option_name, default_value)
        )

    def get_optional_float(
        self, option_name: str, default_value: float | None = None
    ) -> float | None:
        return self._with_scan_error(
            lambda: self._inner.get_optional_float(option_name, default_value)
        )

    def get_bool(self, option_name: str, default_value: bool) -> bool:
        return self._with_scan_error(
            lambda: self._inner.get_bool(option_name, default_value)
        )

    def get_required_string(self, option_name: str) -> str:
        return self._with_scan_error(
            lambda: self._inner.get_required_string(option_name)
        )

    def get_optional_string(
        self, option_name: str, default_value: str | None = None
    ) -> str | None:
        return self._with_scan_error(
            lambda: self._inner.get_optional_string(option_name, default_value)
        )

    def get_mapping(
        self, option_name: str, default_value: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        return self._with_scan_error(
            lambda: self._inner.get_mapping(option_name, default_value)
        )

    def get_string_mapping(self, option_name: str) -> Mapping[str, str]:
        return self._with_scan_error(
            lambda: self._inner.get_string_mapping(option_name)
        )

    def get_path(self, *option_names: str) -> Path:
        return self._with_scan_error(lambda: self._inner.get_path(*option_names))
