from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.llm.providers.errors import ProviderConfigError


def provider_accessor(config: Mapping[str, Any], *, prefix: str) -> ConfigAccessor:
    return ConfigAccessor(config, prefix=prefix, error_type=ProviderConfigError)


@dataclass(frozen=True, slots=True)
class ProviderEndpointConfig:
    model: str
    base_url: str
    endpoint: str
    timeout: float | None

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("model is required")
        if not self.base_url:
            raise ProviderConfigError("base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("endpoint is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("timeout must be greater than 0")


def endpoint_config_from_mapping(
    config: Mapping[str, Any],
    *,
    prefix: str,
    default_timeout: float | None,
) -> ProviderEndpointConfig:
    accessor = provider_accessor(config, prefix=prefix)
    return ProviderEndpointConfig(
        model=accessor.get_required_string("model"),
        base_url=accessor.get_required_string("base_url").rstrip("/"),
        endpoint=accessor.get_required_string("endpoint"),
        timeout=accessor.get_optional_float("timeout", default_timeout),
    )
