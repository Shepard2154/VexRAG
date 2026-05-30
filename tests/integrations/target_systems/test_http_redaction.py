"""HTTP target adapter contract (integration / no network)."""

import pytest

from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
    TargetSystemQuery,
)


class _Resp:
    status = 200

    def read(self) -> bytes:
        return b'{"answer":"ok","contexts":[]}'

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _PayloadResp:
    status = 200

    def read(self) -> bytes:
        return b'{"answer":"x","contexts":["ctx"]}'

    def __enter__(self) -> "_PayloadResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TestHTTPTargetRedaction:
    @pytest.mark.integration
    def test_http_target_metadata_omits_payload_by_default(self) -> None:
        cfg = HTTPTargetSystemAdapterConfig(
            base_url="http://example.invalid",
            include_raw_response_in_metadata=False,
        )

        def _fake_open(req: object, timeout: object | None = None) -> _Resp:
            return _Resp()

        adapter = HTTPTargetSystemAdapter(cfg, http_open=_fake_open)
        reply = adapter.answer(TargetSystemQuery(query="q"))

        assert "response_payload" not in reply.metadata
        assert reply.metadata.get("http_status") == 200

    @pytest.mark.integration
    def test_http_target_metadata_includes_payload_when_enabled(self) -> None:
        cfg = HTTPTargetSystemAdapterConfig(
            base_url="http://example.invalid",
            include_raw_response_in_metadata=True,
        )

        def _fake_open(req: object, timeout: object | None = None) -> _PayloadResp:
            return _PayloadResp()

        adapter = HTTPTargetSystemAdapter(cfg, http_open=_fake_open)
        reply = adapter.answer(TargetSystemQuery(query="q"))

        payload = reply.metadata.get("response_payload")
        assert isinstance(payload, dict)
        assert payload.get("answer") == "x"
