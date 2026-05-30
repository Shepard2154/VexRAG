import pytest

from vexrag.core.llm.providers import ollama
from vexrag.core.llm.providers.config import ProviderEndpointConfig
from vexrag.core.llm.providers.errors import ProviderServiceError


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


def _stub_provider_response(monkeypatch, response) -> None:
    monkeypatch.setattr(ollama, "post_json", lambda **_kwargs: response)


class TestOllamaEmbeddingClient:
    def test_ollama_embedding_client_parses_embeddings(self, monkeypatch) -> None:
        _stub_provider_response(
            monkeypatch,
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
        self, monkeypatch, response
    ) -> None:
        _stub_provider_response(monkeypatch, response)
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

    def test_ollama_embedding_client_rejects_invalid_embedding_items(
        self, monkeypatch
    ) -> None:
        _stub_provider_response(monkeypatch, {"embeddings": [["1", 2]]})
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


class TestOllamaJsonCompletionClient:
    def test_ollama_judge_client_parses_response(self, monkeypatch) -> None:
        _stub_provider_response(monkeypatch, {"response": '{"score": 1}'})
        client = ollama.OllamaJsonCompletionClient(
            _ollama_endpoint(
                model="judge-model",
                endpoint="/api/generate",
            )
        )

        assert client.complete_json("judge this") == '{"score": 1}'

    @pytest.mark.parametrize("response", [{}, {"response": {"score": 1}}])
    def test_ollama_judge_client_requires_string_response(
        self, monkeypatch, response
    ) -> None:
        _stub_provider_response(monkeypatch, response)
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
