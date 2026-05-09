from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ScanConfigError

__all__ = [
    "bool_option",
    "float_option",
    "int_option",
    "mapping_option",
    "optional_float_option",
    "optional_int",
    "optional_string",
    "path_option",
    "required_string",
    "string_mapping_option",
]


def required_string(config: Mapping[str, Any], key: str, prefix: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScanConfigError(f"{prefix}.{key} is required")
    return value.strip()


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ScanConfigError("optional string values must be strings")
    stripped = value.strip()
    return stripped or None


def mapping_option(
    config: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = config.get(key, default)
    if not isinstance(value, Mapping):
        raise ScanConfigError(f"{prefix}.{key} must be a mapping")
    return value


def string_mapping_option(
    config: Mapping[str, Any],
    key: str,
    prefix: str,
) -> Mapping[str, str]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ScanConfigError(f"{prefix}.{key} must be a mapping")
    return {str(name): str(item) for name, item in value.items()}


def int_option(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ScanConfigError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{key} must be an integer") from exc


def optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{name} must be an integer") from exc


def bool_option(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ScanConfigError(f"{key} must be a boolean")
    return value


def float_option(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ScanConfigError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{key} must be a number") from exc


def optional_float_option(
    config: Mapping[str, Any],
    key: str,
    default: float,
) -> float | None:
    value = config.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{key} must be a number or null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{key} must be a number or null") from exc


def path_option(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
    prefix: str,
    *,
    base_dir: Path | None,
) -> Path:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value.strip())
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            return path
    expected = ", ".join(keys)
    raise ScanConfigError(f"{prefix} must configure one of: {expected}")
