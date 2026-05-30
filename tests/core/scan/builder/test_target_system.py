import json
from urllib.request import Request

import pytest

from vexrag.core.scan.builder.target_system import (
    build_target_system,
    create_default_target_system_registry,
)
from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.core.target_systems import HTTPTargetSystemAdapter, TargetSystemQuery


class _JsonResp:
    status = 200

    def read(self) -> bytes:
        return b'{"answer":"from-builder","contexts":["x"]}'

    def __enter__(self) -> "_JsonResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TestBuildTargetSystem:
    def test_build_target_system_from_smoke_yaml_fragment(self) -> None:
        captured: dict[str, object] = {}

        def fake_open(req: Request, timeout: object | None = None) -> _JsonResp:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _JsonResp()

        config = {
            "target_system": {
                "http": {
                    "base_url": "http://localhost:8080",
                    "route": "/model/context-based-response",
                    "method": "POST",
                    "timeout": 60,
                    "request_template": {
                        "query": "{query}",
                        "contexts": "{contexts}",
                    },
                    "response_paths": {
                        "answer": "answer",
                        "contexts": "contexts",
                    },
                },
            },
        }
        adapter = build_target_system(
            config,
            registry=create_default_target_system_registry(),
        )
        assert isinstance(adapter, HTTPTargetSystemAdapter)

        adapter._http_open = fake_open  # type: ignore[method-assign]
        reply = adapter.answer(TargetSystemQuery(query="test", contexts=("ctx",)))

        assert captured["url"] == "http://localhost:8080/model/context-based-response"
        assert captured["body"] == {"query": "test", "contexts": ["ctx"]}
        assert reply.answer == "from-builder"
        assert reply.contexts == ("x",)

    def test_build_target_system_rejects_missing_section(self) -> None:
        with pytest.raises(ScanConfigError, match="target_system must be configured"):
            build_target_system({}, registry=create_default_target_system_registry())

    def test_build_target_system_rejects_non_mapping_http(self) -> None:
        config = {"target_system": {"http": "not-a-mapping"}}
        with pytest.raises(
            ScanConfigError, match="target_system.http must be a mapping"
        ):
            build_target_system(
                config,
                registry=create_default_target_system_registry(),
            )
