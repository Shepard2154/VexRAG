from urllib.error import HTTPError

import pytest

from vexrag.core.llm.providers import http as http_common
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.llm.providers.http import coerce_embedding, post_json


class _FailingReadResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        raise TimeoutError("timed out")


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


class _FailingBody:
    def read(self):
        raise OSError("body unavailable")

    def close(self) -> None:
        pass


def _post_json() -> None:
    post_json(
        base_url="http://example.test",
        endpoint="/api",
        payload={"input": "hello"},
        timeout=1.0,
        service_name="Test provider",
    )


class TestPostJson:
    def test_post_json_maps_response_read_timeout(self, monkeypatch) -> None:
        monkeypatch.setattr(
            http_common,
            "urlopen",
            lambda _request, *, timeout=None: _FailingReadResponse(),
        )

        with pytest.raises(
            ProviderServiceError,
            match="Test provider request failed: timed out",
        ):
            _post_json()

    def test_post_json_maps_urlopen_os_error(self, monkeypatch) -> None:
        def raise_os_error(_request, *, timeout=None):
            raise OSError("connection reset")

        monkeypatch.setattr(http_common, "urlopen", raise_os_error)

        with pytest.raises(
            ProviderServiceError,
            match="Test provider request failed: connection reset",
        ):
            _post_json()

    def test_post_json_maps_request_setup_value_error(self, monkeypatch) -> None:
        def raise_value_error(*_args, **_kwargs):
            raise ValueError("unknown url type")

        monkeypatch.setattr(http_common, "Request", raise_value_error)

        with pytest.raises(
            ProviderServiceError,
            match="Test provider request failed: unknown url type",
        ):
            _post_json()

    def test_post_json_maps_invalid_utf8_response_body(self, monkeypatch) -> None:
        monkeypatch.setattr(
            http_common,
            "urlopen",
            lambda _request, *, timeout=None: _Response(b"\xff"),
        )

        with pytest.raises(
            ProviderServiceError,
            match="Test provider response was not valid UTF-8",
        ):
            _post_json()

    def test_post_json_maps_http_error_body_read_failure(self, monkeypatch) -> None:
        error = HTTPError(
            url="http://example.test/api",
            code=500,
            msg="server error",
            hdrs=None,
            fp=_FailingBody(),
        )

        def raise_http_error(_request, *, timeout=None):
            raise error

        monkeypatch.setattr(http_common, "urlopen", raise_http_error)

        with pytest.raises(
            ProviderServiceError,
            match=(
                "Test provider request returned HTTP 500: "
                "failed to read error body: body unavailable"
            ),
        ):
            _post_json()


class TestCoerceEmbedding:
    def test_coerce_embedding_accepts_numeric_list(self) -> None:
        assert coerce_embedding([1, 2.5]) == (1.0, 2.5)

    @pytest.mark.parametrize(
        "vector",
        [
            ("1", 2),
            ["1", 2],
            [True, 2],
        ],
    )
    def test_coerce_embedding_rejects_non_numeric_items(self, vector) -> None:
        with pytest.raises(
            ProviderServiceError,
            match="embedding response items must be numeric lists",
        ):
            coerce_embedding(vector)

    @pytest.mark.parametrize("vector", [[float("nan")], [float("inf")]])
    def test_coerce_embedding_rejects_non_finite_values(self, vector) -> None:
        with pytest.raises(
            ProviderServiceError,
            match="embedding values must be finite numbers",
        ):
            coerce_embedding(vector)
