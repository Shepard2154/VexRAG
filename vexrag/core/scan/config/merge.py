from collections.abc import Mapping
from typing import Any


def deep_merge_mappings(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged
