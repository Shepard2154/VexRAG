from collections.abc import Mapping
from typing import Any


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new mapping: ``override`` values win; nested dicts are merged recursively."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
            and not isinstance(value, str)
        ):
            result[key] = deep_merge_mappings(result[key], value)
        else:
            result[key] = value
    return result
