from collections.abc import Mapping
from typing import Any

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
    TargetSystemAdapter,
)
from vexrag.core.target_systems.registry import TargetSystemRegistry


def create_default_target_system_registry() -> TargetSystemRegistry:
    return TargetSystemRegistry({"http": build_http_target_system})


def build_target_system(
    config: Mapping[str, Any],
    *,
    registry: TargetSystemRegistry,
) -> TargetSystemAdapter:
    target_config = target_system_section(config)
    transport = str(target_config.get("transport", "http")).strip()
    return registry.get(transport)(config)


def build_http_target_system(config: Mapping[str, Any]) -> HTTPTargetSystemAdapter:
    target_config = target_system_section(config)
    http_config = target_config.get("http", target_config)
    if not isinstance(http_config, Mapping):
        raise ScanConfigError("target_system.http must be a mapping")
    http_config_accessor = ConfigAccessor(
        http_config,
        prefix="target_system.http",
        error_type=ScanConfigError,
    )

    scan_raw = config.get("scan", {})
    include_raw_payload = False
    if isinstance(scan_raw, Mapping):
        scan_accessor = ConfigAccessor(
            scan_raw,
            prefix="scan",
            error_type=ScanConfigError,
        )
        include_raw_payload = scan_accessor.get_bool(
            "debug_include_raw_target_response",
            False,
        )

    return HTTPTargetSystemAdapter(
        HTTPTargetSystemAdapterConfig(
            base_url=http_config_accessor.get_required_string("base_url"),
            route=str(http_config.get("route", "")).strip(),
            method=str(http_config.get("method", "POST")).strip(),
            timeout=http_config_accessor.get_optional_float("timeout", 10.0),
            request_template=http_config_accessor.get_mapping(
                "request_template",
                {"query": "{query}", "contexts": "{contexts}"},
            ),
            response_paths=http_config_accessor.get_mapping(
                "response_paths",
                {"answer": "answer", "contexts": "contexts"},
            ),
            headers=http_config_accessor.get_string_mapping("headers"),
            include_raw_response_in_metadata=include_raw_payload,
        )
    )


def target_system_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    target_config = config.get("target_system")
    if not isinstance(target_config, Mapping):
        raise ScanConfigError("target_system must be configured")
    return target_config
