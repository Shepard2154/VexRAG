from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vexrag.core.target_systems.contracts import TargetSystemAdapter

TargetSystemBuilder = Callable[[Mapping[str, Any]], TargetSystemAdapter]


@dataclass(frozen=True, slots=True)
class TargetSystemRegistry:
    _builders: Mapping[str, TargetSystemBuilder]

    def get(self, transport: str) -> TargetSystemBuilder:
        key = transport.strip()
        try:
            return self._builders[key]
        except KeyError as err:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(
                f"unknown target system transport {transport!r}; supported: {supported}"
            ) from err
