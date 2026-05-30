"""HTTP target adapter transport contract (unit, mocked http_open)."""

import json
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from vexrag.core.target_systems import (
    HTTPTargetSystemAdapter,
    HTTPTargetSystemAdapterConfig,
    TargetSystemAdapterError,
    TargetSystemQuery,
)


class _JsonResp:
    def __init__(self, payload: dict, *, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_JsonResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TestHTTPTargetAdapter:
    def test_post_sends_json_body_and_parses_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_open(req: Request, timeout: object | None = None) -> _JsonResp:
            captured["method"] = req.method
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _JsonResp(
                {"answer": "yes", "contexts": ["ctx-a", "ctx-b"]},
            )

        cfg = HTTPTargetSystemAdapterConfig(
            base_url="http://example.invalid",
            route="api/query",
            method="POST",
            timeout=15.0,
        )
        adapter = HTTPTargetSystemAdapter(cfg, http_open=fake_open)
        reply = adapter.answer(
            TargetSystemQuery(query="what?", contexts=("c1",)),
        )

        assert captured["method"] == "POST"
        assert captured["url"] == "http://example.invalid/api/query"
        assert captured["body"] == {"query": "what?", "contexts": ["c1"]}
        assert captured["timeout"] == 15.0
        assert reply.answer == "yes"
        assert reply.contexts == ("ctx-a", "ctx-b")

    def test_get_encodes_query_params_in_url(self) -> None:
        captured: dict[str, object] = {}

        def fake_open(req: Request, timeout: object | None = None) -> _JsonResp:
            captured["method"] = req.method
            captured["url"] = req.full_url
            captured["data"] = req.data
            return _JsonResp({"answer": "ok", "contexts": []})

        cfg = HTTPTargetSystemAdapterConfig(
            base_url="http://example.invalid",
            route="search",
            method="GET",
            request_template={"q": "{query}", "k": "3"},
        )
        adapter = HTTPTargetSystemAdapter(cfg, http_open=fake_open)
        reply = adapter.answer(TargetSystemQuery(query="hello world"))

        assert captured["method"] == "GET"
        assert captured["data"] is None
        url = str(captured["url"])
        assert "q=hello+world" in url or "q=hello%20world" in url
        assert "k=3" in url
        assert reply.answer == "ok"

    def test_nested_response_paths_extract_answer(self) -> None:
        cfg = HTTPTargetSystemAdapterConfig(
            base_url="http://example.invalid",
            response_paths={"answer": "data.answer", "contexts": "data.items"},
        )

        def fake_open(req: object, timeout: object | None = None) -> _JsonResp:
            return _JsonResp(
                {
                    "data": {
                        "answer": "nested",
                        "items": ["a", "b"],
                    },
                },
            )

        adapter = HTTPTargetSystemAdapter(cfg, http_open=fake_open)
        reply = adapter.answer(TargetSystemQuery(query="q"))
        assert reply.answer == "nested"
        assert reply.contexts == ("a", "b")

    @pytest.mark.parametrize(
        ("status_code", "body"),
        [(500, b"internal error"), (404, b"not found")],
    )
    def test_http_error_raises_adapter_error(
        self, status_code: int, body: bytes
    ) -> None:
        def fake_open(req: object, timeout: object | None = None) -> None:
            raise HTTPError(
                url="http://example.invalid/",
                code=status_code,
                msg="error",
                hdrs=None,
                fp=BytesIO(body),
            )

        adapter = HTTPTargetSystemAdapter(
            HTTPTargetSystemAdapterConfig(base_url="http://example.invalid"),
            http_open=fake_open,
        )
        with pytest.raises(TargetSystemAdapterError, match=f"HTTP {status_code}"):
            adapter.answer(TargetSystemQuery(query="q"))

    def test_url_error_raises_adapter_error(self) -> None:
        def fake_open(req: object, timeout: object | None = None) -> None:
            raise URLError("connection refused")

        adapter = HTTPTargetSystemAdapter(
            HTTPTargetSystemAdapterConfig(base_url="http://example.invalid"),
            http_open=fake_open,
        )
        with pytest.raises(TargetSystemAdapterError, match="connection refused"):
            adapter.answer(TargetSystemQuery(query="q"))

    def test_non_json_body_raises_adapter_error(self) -> None:
        class _TextResp:
            status = 200

            def read(self) -> bytes:
                return b"not json"

            def __enter__(self) -> "_TextResp":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        adapter = HTTPTargetSystemAdapter(
            HTTPTargetSystemAdapterConfig(base_url="http://example.invalid"),
            http_open=lambda *_a, **_k: _TextResp(),
        )
        with pytest.raises(TargetSystemAdapterError, match="non-JSON"):
            adapter.answer(TargetSystemQuery(query="q"))

    def test_missing_answer_path_raises_adapter_error(self) -> None:
        adapter = HTTPTargetSystemAdapter(
            HTTPTargetSystemAdapterConfig(base_url="http://example.invalid"),
            http_open=lambda *_a, **_k: _JsonResp({"answer": None, "contexts": []}),
        )
        with pytest.raises(TargetSystemAdapterError, match="null"):
            adapter.answer(TargetSystemQuery(query="q"))
