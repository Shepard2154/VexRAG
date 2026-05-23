from urllib.error import HTTPError

import pytest

from vexrag.core.llm.providers import http as http_common
from vexrag.core.llm.providers import ollama, vllm
from vexrag.core.llm.providers.config import ProviderEndpointConfig
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.llm.providers.http import coerce_embedding, post_json
from vexrag.core.llm.providers.vllm import (
    VLLMEndpointConfig,
)
from vexrag.core.llm.providers.vllm import (
    build_embedding_client as build_vllm_embedding_client,
)
from vexrag.core.llm.providers.vllm import (
    build_json_completion_client as build_vllm_json_completion_client,
)

_EXPECTED_VLLM_API_KEY = "vllm-local"


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


def _ollama_endpoint(
    *,
    model: str,
    base_url: str = "http://localhost:11434",
    endpoint: str,
) -> ProviderEndpointConfig:
    return ProviderEndpointConfig(
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        timeout=30.0,
    )


def _vllm_endpoint(
    *,
    model: str,
    base_url: str = "http://localhost:8000",
    endpoint: str,
) -> VLLMEndpointConfig:
    return VLLMEndpointConfig(
        endpoint=ProviderEndpointConfig(
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            timeout=30.0,
        )
    )


def _post_json() -> None:
    post_json(
        base_url="http://example.test",
        endpoint="/api",
        payload={"input": "hello"},
        timeout=1.0,
        service_name="Test provider",
    )


def _stub_provider_response(monkeypatch, provider_module, response) -> None:
    monkeypatch.setattr(provider_module, "post_json", lambda **_kwargs: response)


def test_post_json_maps_response_read_timeout(monkeypatch) -> None:
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


def test_post_json_maps_urlopen_os_error(monkeypatch) -> None:
    def raise_os_error(_request, *, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(http_common, "urlopen", raise_os_error)

    with pytest.raises(
        ProviderServiceError,
        match="Test provider request failed: connection reset",
    ):
        _post_json()


def test_post_json_maps_request_setup_value_error(monkeypatch) -> None:
    def raise_value_error(*_args, **_kwargs):
        raise ValueError("unknown url type")

    monkeypatch.setattr(http_common, "Request", raise_value_error)

    with pytest.raises(
        ProviderServiceError,
        match="Test provider request failed: unknown url type",
    ):
        _post_json()


def test_post_json_maps_invalid_utf8_response_body(monkeypatch) -> None:
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


def test_post_json_maps_http_error_body_read_failure(monkeypatch) -> None:
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


def test_coerce_embedding_accepts_numeric_list() -> None:
    assert coerce_embedding([1, 2.5]) == (1.0, 2.5)


@pytest.mark.parametrize(
    "vector",
    [
        ("1", 2),
        ["1", 2],
        [True, 2],
    ],
)
def test_coerce_embedding_rejects_non_numeric_items(vector) -> None:
    with pytest.raises(
        ProviderServiceError,
        match="embedding response items must be numeric lists",
    ):
        coerce_embedding(vector)


@pytest.mark.parametrize("vector", [[float("nan")], [float("inf")]])
def test_coerce_embedding_rejects_non_finite_values(vector) -> None:
    with pytest.raises(
        ProviderServiceError,
        match="embedding values must be finite numbers",
    ):
        coerce_embedding(vector)


def test_ollama_embedding_client_parses_embeddings(monkeypatch) -> None:
    _stub_provider_response(
        monkeypatch,
        ollama,
        {"embeddings": [[1, 2.5], [3.0, 4]]},
    )
    client = ollama.OllamaEmbeddingClient(
        _ollama_endpoint(
            model="embedding-model",
            endpoint="/api/embed",
        )
    )

    assert client.embed_texts(["one", "two"]) == ((1.0, 2.5), (3.0, 4.0))


@pytest.mark.parametrize("response", [{}, {"embeddings": "not-a-list"}])
def test_ollama_embedding_client_requires_embeddings_list(
    monkeypatch, response
) -> None:
    _stub_provider_response(monkeypatch, ollama, response)
    client = ollama.OllamaEmbeddingClient(
        _ollama_endpoint(
            model="embedding-model",
            endpoint="/api/embed",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="Ollama embedding response must include an 'embeddings' list",
    ):
        client.embed_texts(["text"])


def test_ollama_embedding_client_rejects_invalid_embedding_items(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, ollama, {"embeddings": [["1", 2]]})
    client = ollama.OllamaEmbeddingClient(
        _ollama_endpoint(
            model="embedding-model",
            endpoint="/api/embed",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="embedding response items must be numeric lists",
    ):
        client.embed_texts(["text"])


def test_ollama_judge_client_parses_response(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, ollama, {"response": '{"score": 1}'})
    client = ollama.OllamaJsonCompletionClient(
        _ollama_endpoint(
            model="judge-model",
            endpoint="/api/generate",
        )
    )

    assert client.complete_json("judge this") == '{"score": 1}'


@pytest.mark.parametrize("response", [{}, {"response": {"score": 1}}])
def test_ollama_judge_client_requires_string_response(monkeypatch, response) -> None:
    _stub_provider_response(monkeypatch, ollama, response)
    client = ollama.OllamaJsonCompletionClient(
        _ollama_endpoint(
            model="judge-model",
            endpoint="/api/generate",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="Ollama judge response must include a string 'response' field",
    ):
        client.complete_json("judge this")


def test_vllm_embedding_client_parses_data_embeddings(monkeypatch) -> None:
    _stub_provider_response(
        monkeypatch,
        vllm,
        {"data": [{"embedding": [1, 2.5]}, {"embedding": [3.0, 4]}]},
    )
    client = vllm.VLLMEmbeddingClient(
        _vllm_endpoint(
            model="embedding-model",
            endpoint="/v1/embeddings",
        )
    )

    assert client.embed_texts(["one", "two"]) == ((1.0, 2.5), (3.0, 4.0))


@pytest.mark.parametrize("response", [{}, {"data": "not-a-list"}])
def test_vllm_embedding_client_requires_data_list(monkeypatch, response) -> None:
    _stub_provider_response(monkeypatch, vllm, response)
    client = vllm.VLLMEmbeddingClient(
        _vllm_endpoint(
            model="embedding-model",
            endpoint="/v1/embeddings",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM embedding response must include a 'data' list",
    ):
        client.embed_texts(["text"])


def test_vllm_embedding_client_requires_object_items(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, vllm, {"data": [[1, 2]]})
    client = vllm.VLLMEmbeddingClient(
        _vllm_endpoint(
            model="embedding-model",
            endpoint="/v1/embeddings",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM embedding response items must be objects",
    ):
        client.embed_texts(["text"])


def test_vllm_embedding_client_requires_numeric_embedding(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, vllm, {"data": [{"embedding": ["1", 2]}]})
    client = vllm.VLLMEmbeddingClient(
        _vllm_endpoint(
            model="embedding-model",
            endpoint="/v1/embeddings",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="embedding response items must be numeric lists",
    ):
        client.embed_texts(["text"])


def test_vllm_judge_client_parses_message_content(monkeypatch) -> None:
    _stub_provider_response(
        monkeypatch,
        vllm,
        {"choices": [{"message": {"content": '{"score": 1}'}}]},
    )
    client = vllm.VLLMJsonCompletionClient(
        _vllm_endpoint(
            model="judge-model",
            endpoint="/v1/chat/completions",
        )
    )

    assert client.complete_json("judge this") == '{"score": 1}'


@pytest.mark.parametrize("response", [{}, {"choices": []}, {"choices": "none"}])
def test_vllm_judge_client_requires_non_empty_choices(monkeypatch, response) -> None:
    _stub_provider_response(monkeypatch, vllm, response)
    client = vllm.VLLMJsonCompletionClient(
        _vllm_endpoint(
            model="judge-model",
            endpoint="/v1/chat/completions",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM judge response must include a non-empty 'choices' list",
    ):
        client.complete_json("judge this")


def test_vllm_judge_client_requires_object_choice(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, vllm, {"choices": ["not-an-object"]})
    client = vllm.VLLMJsonCompletionClient(
        _vllm_endpoint(
            model="judge-model",
            endpoint="/v1/chat/completions",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM judge choice items must be objects",
    ):
        client.complete_json("judge this")


def test_vllm_judge_client_requires_message_object(monkeypatch) -> None:
    _stub_provider_response(monkeypatch, vllm, {"choices": [{"message": "none"}]})
    client = vllm.VLLMJsonCompletionClient(
        _vllm_endpoint(
            model="judge-model",
            endpoint="/v1/chat/completions",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM judge response choice must include a 'message' object",
    ):
        client.complete_json("judge this")


def test_vllm_judge_client_requires_string_content(monkeypatch) -> None:
    _stub_provider_response(
        monkeypatch,
        vllm,
        {"choices": [{"message": {"content": {"score": 1}}}]},
    )
    client = vllm.VLLMJsonCompletionClient(
        _vllm_endpoint(
            model="judge-model",
            endpoint="/v1/chat/completions",
        )
    )

    with pytest.raises(
        ProviderServiceError,
        match="vLLM judge response message must include string 'content'",
    ):
        client.complete_json("judge this")


@pytest.mark.parametrize("config_api_key", [None, "", "   "])
def test_vllm_embedding_client_defaults_blank_api_key(config_api_key) -> None:
    config = {
        "model": "embedding-model",
        "base_url": "http://localhost:8000/",
        "endpoint": "/v1/embeddings",
    }
    if config_api_key is not None:
        config["api_key"] = config_api_key

    client = build_vllm_embedding_client(config)

    assert client.api_key == _EXPECTED_VLLM_API_KEY


def test_vllm_embedding_client_strips_api_key() -> None:
    client = build_vllm_embedding_client(
        {
            "model": "embedding-model",
            "base_url": "http://localhost:8000/",
            "endpoint": "/v1/embeddings",
            "api_key": "  custom-key  ",
        }
    )

    assert client.api_key == "custom-key"


@pytest.mark.parametrize("config_api_key", [None, "", "   "])
def test_vllm_judge_client_defaults_blank_api_key(config_api_key) -> None:
    config = {
        "model": "judge-model",
        "base_url": "http://localhost:8000/",
        "endpoint": "/v1/chat/completions",
    }
    if config_api_key is not None:
        config["api_key"] = config_api_key

    client = build_vllm_json_completion_client(config)

    assert client.api_key == _EXPECTED_VLLM_API_KEY


def test_vllm_judge_client_strips_api_key() -> None:
    client = build_vllm_json_completion_client(
        {
            "model": "judge-model",
            "base_url": "http://localhost:8000/",
            "endpoint": "/v1/chat/completions",
            "api_key": "  custom-key  ",
        }
    )

    assert client.api_key == "custom-key"
