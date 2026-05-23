from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from vexrag.core.exceptions import ConfigurationError

_Number = TypeVar("_Number", float, int)


def _reject_bool(
    value: Any, error_message: str, error_factory: Callable[[str], Exception]
) -> None:
    if isinstance(value, bool):
        raise error_factory(error_message)


def _safe_cast(
    value: Any,
    cast_to: type,
    error_message: str,
    error_factory: Callable[[str], Exception],
) -> Any:
    try:
        return cast_to(value)
    except (TypeError, ValueError) as err:
        raise error_factory(error_message) from err


class ConfigAccessor:
    def __init__(
        self,
        config: Mapping[str, Any],
        prefix: str = "",
        base_dir: Path | None = None,
        error_type: type[Exception] = ConfigurationError,
    ) -> None:
        self.config = config
        self.prefix = prefix
        self.base_dir = base_dir
        self._error_type = error_type

    def _error(self, message: str) -> Exception:
        return self._error_type(message)

    def _field_path(self, option_name: str) -> str:
        return f"{self.prefix}.{option_name}" if self.prefix else option_name

    def _cast_number(
        self, raw_value: Any, cast_to: type[_Number], field_path: str
    ) -> _Number:
        message_on_type_mismatch = f"{field_path} must be {cast_to.__name__}"
        _reject_bool(raw_value, message_on_type_mismatch, self._error)
        return _safe_cast(
            raw_value,
            cast_to,
            message_on_type_mismatch,
            self._error,
        )

    def _get_number(
        self, option_name: str, default_value: _Number, cast_to: type[_Number]
    ) -> _Number:
        raw_value = self.config.get(option_name, default_value)
        field_path = self._field_path(option_name)
        return self._cast_number(raw_value, cast_to, field_path)

    def _get_optional_number(
        self, option_name: str, default_value: _Number | None, cast_to: type[_Number]
    ) -> _Number | None:
        raw_value = self.config.get(option_name, default_value)
        if raw_value is None:
            return None
        field_path = self._field_path(option_name)
        return self._cast_number(raw_value, cast_to, field_path)

    def get_int(self, option_name: str, default_value: int) -> int:
        return self._get_number(option_name, default_value, int)

    def get_float(self, option_name: str, default_value: float) -> float:
        return self._get_number(option_name, default_value, float)

    def get_optional_int(
        self, option_name: str, default_value: int | None = None
    ) -> int | None:
        return self._get_optional_number(option_name, default_value, int)

    def get_optional_float(
        self, option_name: str, default_value: float | None = None
    ) -> float | None:
        return self._get_optional_number(option_name, default_value, float)

    def get_bool(self, option_name: str, default_value: bool) -> bool:
        raw_value = self.config.get(option_name, default_value)
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, bool):
            raise self._error(f"{field_path} must be a boolean")
        return raw_value

    def get_required_string(self, option_name: str) -> str:
        raw_value = self.config.get(option_name)
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise self._error(f"{field_path} must be a non-empty string")
        return raw_value.strip()

    def get_optional_string(
        self, option_name: str, default_value: str | None = None
    ) -> str | None:
        raw_value = self.config.get(option_name, default_value)
        if raw_value is None:
            return None
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, str):
            raise self._error(f"{field_path} must be a string")
        stripped = raw_value.strip()
        return stripped or None

    def get_mapping(
        self, option_name: str, default_value: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        raw_value = self.config.get(option_name, default_value)
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, Mapping):
            raise self._error(f"{field_path} must be a mapping")
        return raw_value

    def get_string_mapping(self, option_name: str) -> Mapping[str, str]:
        raw_value = self.config.get(option_name, {})
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, Mapping):
            raise self._error(f"{field_path} must be a mapping")
        return {str(name): str(item) for name, item in raw_value.items()}

    def get_path(self, *option_names: str) -> Path:
        for key in option_names:
            raw_value = self.config.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                path = Path(raw_value.strip())
                if not path.is_absolute() and self.base_dir is not None:
                    path = self.base_dir / path
                return path
        expected = ", ".join(option_names)
        raise self._error(f"{self.prefix} must configure one of: {expected}")
