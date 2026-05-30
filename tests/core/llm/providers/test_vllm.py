import pytest

from vexrag.core.llm.providers import vllm
from vexrag.core.llm.providers.config import ProviderEndpointConfig
from vexrag.core.llm.providers.errors import ProviderServiceError
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


def _stub_provider_response(monkeypatch, response) -> None:
    monkeypatch.setattr(vllm, "post_json", lambda **_kwargs: response)


class TestVLLMEmbeddingClient:
    def test_vllm_embedding_client_parses_data_embeddings(self, monkeypatch) -> None:
        _stub_provider_response(
            monkeypatch,
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
    def test_vllm_embedding_client_requires_data_list(
        self, monkeypatch, response
    ) -> None:
        _stub_provider_response(monkeypatch, response)
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

    def test_vllm_embedding_client_requires_object_items(self, monkeypatch) -> None:
        _stub_provider_response(monkeypatch, {"data": [[1, 2]]})
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

    def test_vllm_embedding_client_requires_numeric_embedding(
        self, monkeypatch
    ) -> None:
        _stub_provider_response(monkeypatch, {"data": [{"embedding": ["1", 2]}]})
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

    @pytest.mark.parametrize("config_api_key", [None, "", "   "])
    def test_vllm_embedding_client_defaults_blank_api_key(self, config_api_key) -> None:
        config = {
            "model": "embedding-model",
            "base_url": "http://localhost:8000/",
            "endpoint": "/v1/embeddings",
        }
        if config_api_key is not None:
            config["api_key"] = config_api_key

        client = build_vllm_embedding_client(config)

        assert client.api_key == _EXPECTED_VLLM_API_KEY

    def test_vllm_embedding_client_strips_api_key(self) -> None:
        client = build_vllm_embedding_client(
            {
                "model": "embedding-model",
                "base_url": "http://localhost:8000/",
                "endpoint": "/v1/embeddings",
                "api_key": "  custom-key  ",
            }
        )

        assert client.api_key == "custom-key"


class TestVLLMJsonCompletionClient:
    def test_vllm_judge_client_parses_message_content(self, monkeypatch) -> None:
        _stub_provider_response(
            monkeypatch,
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
    def test_vllm_judge_client_requires_non_empty_choices(
        self, monkeypatch, response
    ) -> None:
        _stub_provider_response(monkeypatch, response)
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

    def test_vllm_judge_client_requires_object_choice(self, monkeypatch) -> None:
        _stub_provider_response(monkeypatch, {"choices": ["not-an-object"]})
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

    def test_vllm_judge_client_requires_message_object(self, monkeypatch) -> None:
        _stub_provider_response(monkeypatch, {"choices": [{"message": "none"}]})
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

    def test_vllm_judge_client_requires_string_content(self, monkeypatch) -> None:
        _stub_provider_response(
            monkeypatch,
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
    def test_vllm_judge_client_defaults_blank_api_key(self, config_api_key) -> None:
        config = {
            "model": "judge-model",
            "base_url": "http://localhost:8000/",
            "endpoint": "/v1/chat/completions",
        }
        if config_api_key is not None:
            config["api_key"] = config_api_key

        client = build_vllm_json_completion_client(config)

        assert client.api_key == _EXPECTED_VLLM_API_KEY

    def test_vllm_judge_client_strips_api_key(self) -> None:
        client = build_vllm_json_completion_client(
            {
                "model": "judge-model",
                "base_url": "http://localhost:8000/",
                "endpoint": "/v1/chat/completions",
                "api_key": "  custom-key  ",
            }
        )

        assert client.api_key == "custom-key"
