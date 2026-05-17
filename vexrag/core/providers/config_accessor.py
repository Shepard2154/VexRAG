from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from vexrag.core.config_accessor import ConfigAccessor
from vexrag.core.errors import ConfigError
from vexrag.core.providers.errors import ProviderConfigError

_T = TypeVar("_T")


class ProviderConfigAccessor:
    """Adapter that wraps ConfigAccessor and guarantees ProviderConfigError."""

    def __init__(self, config: Mapping[str, Any], prefix: str = "") -> None:
        self._inner = ConfigAccessor(config, prefix=prefix)

    def _with_provider_error(self, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except ConfigError as exc:
            raise ProviderConfigError(str(exc)) from exc

    def get_float(self, option_name: str, default_value: float) -> float:
        return self._with_provider_error(
            lambda: self._inner.get_float(option_name, default_value)
        )

    def get_optional_float(
        self, option_name: str, default_value: float | None = None
    ) -> float | None:
        return self._with_provider_error(
            lambda: self._inner.get_optional_float(option_name, default_value)
        )

    def get_required_string(self, option_name: str) -> str:
        return self._with_provider_error(
            lambda: self._inner.get_required_string(option_name)
        )

    def get_optional_string(
        self, option_name: str, default_value: str | None = None
    ) -> str | None:
        return self._with_provider_error(
            lambda: self._inner.get_optional_string(option_name, default_value)
        )
