from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from .errors import ScanConfigError

_Number = TypeVar("_Number", float, int)


def _reject_bool(value: Any, error_message: str) -> None:
    if isinstance(value, bool):
        raise ScanConfigError(error_message)


def _safe_cast(value: Any, cast_to: type, error_message: str) -> Any:
    try:
        return cast_to(value)
    except (TypeError, ValueError) as err:
        raise ScanConfigError(error_message) from err


class ConfigAccessor:
    def __init__(
        self, config: Mapping[str, Any], prefix: str = "", base_dir: Path | None = None
    ):
        self.config = config
        self.prefix = prefix
        self.base_dir = base_dir

    def _field_path(self, option_name: str):
        return f"{self.prefix}.{option_name}" if self.prefix else option_name

    def _cast_number(
        self, raw_value: Any, cast_to: type[_Number], field_path: str
    ) -> _Number:
        message_on_type_mismatch = f"{field_path} must be {cast_to.__name__}"
        _reject_bool(raw_value, message_on_type_mismatch)
        return _safe_cast(raw_value, cast_to, message_on_type_mismatch)

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
            raise ScanConfigError(f"{field_path} must be a boolean")
        return raw_value

    def get_required_string(self, option_name: str) -> str:
        raw_value = self.config.get(option_name)
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ScanConfigError(f"{field_path} must be a non-empty string")
        return raw_value.strip()

    def get_optional_string(
        self, option_name: str, default_value: str | None = None
    ) -> str | None:
        raw_value = self.config.get(option_name, default_value)
        if raw_value is None:
            return None
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, str):
            raise ScanConfigError(f"{field_path} must be a string")
        stripped = raw_value.strip()
        return stripped or None

    def get_mapping(
        self, option_name: str, default_value: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        raw_value = self.config.get(option_name, default_value)
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, Mapping):
            raise ScanConfigError(f"{field_path} must be a mapping")
        return raw_value

    def get_string_mapping(self, option_name: str) -> Mapping[str, str]:
        raw_value = self.config.get(option_name, {})
        field_path = self._field_path(option_name)
        if not isinstance(raw_value, Mapping):
            raise ScanConfigError(f"{field_path} must be a mapping")
        return {str(name): str(item) for name, item in raw_value.items()}

    def get_path(self, *keys: str) -> Path:
        for key in keys:
            raw_value = self.config.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                path = Path(raw_value.strip())
                if not path.is_absolute() and self.base_dir is not None:
                    path = self.base_dir / path
                return path
        expected = ", ".join(keys)
        raise ScanConfigError(f"{self.prefix} must configure one of: {expected}")


# def required_string(config: Mapping[str, Any], key: str, prefix: str) -> str:
#     value = config.get(key)
#     if not isinstance(value, str) or not value.strip():
#         raise ScanConfigError(f"{prefix}.{key} is required")
#     return value.strip()


# def optional_string(value: Any) -> str | None:
#     if value is None:
#         return None
#     if not isinstance(value, str):
#         raise ScanConfigError("optional string values must be strings")
#     stripped = value.strip()
#     return stripped or None


# def mapping_option(
#     config: Mapping[str, Any],
#     key: str,
#     prefix: str,
#     *,
#     default: Mapping[str, Any],
# ) -> Mapping[str, Any]:
#     value = config.get(key, default)
#     if not isinstance(value, Mapping):
#         raise ScanConfigError(f"{prefix}.{key} must be a mapping")
#     return value


# def string_mapping_option(
#     config: Mapping[str, Any],
#     key: str,
#     prefix: str,
# ) -> Mapping[str, str]:
#     value = config.get(key, {})
#     if not isinstance(value, Mapping):
#         raise ScanConfigError(f"{prefix}.{key} must be a mapping")
#     return {str(name): str(item) for name, item in value.items()}


# def int_option(config: Mapping[str, Any], key: str, default: int) -> int:
#     value = config.get(key, default)
#     if isinstance(value, bool):
#         raise ScanConfigError(f"{key} must be an integer")
#     try:
#         return int(value)
#     except (TypeError, ValueError) as exc:
#         raise ScanConfigError(f"{key} must be an integer") from exc


# def optional_int(value: Any, name: str) -> int | None:
#     if value is None:
#         return None
#     if isinstance(value, bool):
#         raise ScanConfigError(f"{name} must be an integer")
#     try:
#         return int(value)
#     except (TypeError, ValueError) as exc:
#         raise ScanConfigError(f"{name} must be an integer") from exc


# def bool_option(config: Mapping[str, Any], key: str, default: bool) -> bool:
#     value = config.get(key, default)
#     if not isinstance(value, bool):
#         raise ScanConfigError(f"{key} must be a boolean")
#     return value


# def float_option(config: Mapping[str, Any], key: str, default: float) -> float:
#     value = config.get(key, default)
#     if isinstance(value, bool):
#         raise ScanConfigError(f"{key} must be a number")
#     try:
#         return float(value)
#     except (TypeError, ValueError) as exc:
#         raise ScanConfigError(f"{key} must be a number") from exc


# def optional_float_option(
#     config: Mapping[str, Any],
#     key: str,
#     default: float,
# ) -> float | None:
#     value = config.get(key, default)
#     if value is None:
#         return None
#     if isinstance(value, bool):
#         raise ScanConfigError(f"{key} must be a number or null")
#     try:
#         return float(value)
#     except (TypeError, ValueError) as exc:
#         raise ScanConfigError(f"{key} must be a number or null") from exc


# def path_option(
#     config: Mapping[str, Any],
#     keys: tuple[str, ...],
#     prefix: str,
#     *,
#     base_dir: Path | None,
# ) -> Path:
#     for key in keys:
#         value = config.get(key)
#         if isinstance(value, str) and value.strip():
#             path = Path(value.strip())
#             if not path.is_absolute() and base_dir is not None:
#                 path = base_dir / path
#             return path
#     expected = ", ".join(keys)
#     raise ScanConfigError(f"{prefix} must configure one of: {expected}")
