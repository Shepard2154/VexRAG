from collections.abc import Mapping, Sequence
from typing import Any

from vexrag.core.llm.providers.config import (
    ProviderEndpointConfig,
    endpoint_config_from_mapping,
    provider_accessor,
)
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.llm.providers.http import coerce_embedding, post_json


class OllamaEmbeddingClient:
    def __init__(self, config: ProviderEndpointConfig) -> None:
        self._config = config

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def endpoint(self) -> str:
        return self._config.endpoint

    @property
    def timeout(self) -> float | None:
        return self._config.timeout

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        payload = {"model": self.model, "input": list(texts)}
        response = post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            payload=payload,
            timeout=self.timeout,
            service_name="Ollama embedding",
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise ProviderServiceError(
                "Ollama embedding response must include an 'embeddings' list"
            )
        return tuple(coerce_embedding(vector) for vector in embeddings)


class OllamaJsonCompletionClient:
    def __init__(
        self,
        config: ProviderEndpointConfig,
        *,
        temperature: float = 0.0,
    ) -> None:
        self._config = config
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def endpoint(self) -> str:
        return self._config.endpoint

    @property
    def timeout(self) -> float | None:
        return self._config.timeout

    @property
    def temperature(self) -> float:
        return self._temperature

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        response = post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            payload=payload,
            timeout=self.timeout,
            service_name="Ollama judge",
        )
        raw_response = response.get("response")
        if not isinstance(raw_response, str):
            raise ProviderServiceError(
                "Ollama judge response must include a string 'response' field"
            )
        return raw_response


def build_embedding_client(config: Mapping[str, Any]) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(
        endpoint_config_from_mapping(
            config,
            prefix="embedding_client",
            default_timeout=30.0,
        )
    )


def build_json_completion_client(
    config: Mapping[str, Any],
) -> OllamaJsonCompletionClient:
    accessor = provider_accessor(config, prefix="judge_client")
    return OllamaJsonCompletionClient(
        endpoint_config_from_mapping(
            config,
            prefix="judge_client",
            default_timeout=60.0,
        ),
        temperature=accessor.get_float("temperature", 0.0),
    )
