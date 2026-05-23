from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.core.llm.providers.config import (
    ProviderEndpointConfig,
    endpoint_config_from_mapping,
    provider_accessor,
)
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.llm.providers.http import coerce_embedding, post_json

_DEFAULT_API_KEY = "vllm-local"


def _normalize_api_key(api_key: str | None) -> str:
    if api_key is None:
        return _DEFAULT_API_KEY
    stripped = api_key.strip()
    return stripped or _DEFAULT_API_KEY


@dataclass(frozen=True, slots=True)
class VLLMEndpointConfig:
    endpoint: ProviderEndpointConfig
    api_key: str = _DEFAULT_API_KEY

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", _normalize_api_key(self.api_key))


class VLLMEmbeddingClient:
    def __init__(self, config: VLLMEndpointConfig) -> None:
        self._config = config

    @property
    def model(self) -> str:
        return self._config.endpoint.model

    @property
    def base_url(self) -> str:
        return self._config.endpoint.base_url

    @property
    def endpoint(self) -> str:
        return self._config.endpoint.endpoint

    @property
    def timeout(self) -> float | None:
        return self._config.endpoint.timeout

    @property
    def api_key(self) -> str:
        return self._config.api_key

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        payload = {"model": self.model, "input": list(texts)}
        response = post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            api_key=self.api_key,
            payload=payload,
            timeout=self.timeout,
            service_name="vLLM embedding",
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise ProviderServiceError(
                "vLLM embedding response must include a 'data' list"
            )
        vectors = []
        for item in data:
            if not isinstance(item, Mapping):
                raise ProviderServiceError(
                    "vLLM embedding response items must be objects"
                )
            embedding = item.get("embedding")
            vectors.append(coerce_embedding(embedding))
        return tuple(vectors)


class VLLMJsonCompletionClient:
    def __init__(
        self,
        config: VLLMEndpointConfig,
        *,
        temperature: float = 0.0,
    ) -> None:
        self._config = config
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._config.endpoint.model

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def base_url(self) -> str:
        return self._config.endpoint.base_url

    @property
    def endpoint(self) -> str:
        return self._config.endpoint.endpoint

    @property
    def timeout(self) -> float | None:
        return self._config.endpoint.timeout

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def api_key(self) -> str:
        return self._config.api_key

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        response = post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            api_key=self.api_key,
            payload=payload,
            timeout=self.timeout,
            service_name="vLLM judge",
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderServiceError(
                "vLLM judge response must include a non-empty 'choices' list"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise ProviderServiceError("vLLM judge choice items must be objects")
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderServiceError(
                "vLLM judge response choice must include a 'message' object"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderServiceError(
                "vLLM judge response message must include string 'content'"
            )
        return content


def _endpoint_with_api_key(
    config: Mapping[str, Any],
    *,
    prefix: str,
    default_timeout: float | None,
) -> VLLMEndpointConfig:
    accessor = provider_accessor(config, prefix=prefix)
    return VLLMEndpointConfig(
        endpoint=endpoint_config_from_mapping(
            config,
            prefix=prefix,
            default_timeout=default_timeout,
        ),
        api_key=_normalize_api_key(accessor.get_optional_string("api_key")),
    )


def build_embedding_client(config: Mapping[str, Any]) -> VLLMEmbeddingClient:
    return VLLMEmbeddingClient(
        _endpoint_with_api_key(
            config,
            prefix="embedding_client",
            default_timeout=30.0,
        )
    )


def build_json_completion_client(config: Mapping[str, Any]) -> VLLMJsonCompletionClient:
    accessor = provider_accessor(config, prefix="judge_client")
    return VLLMJsonCompletionClient(
        _endpoint_with_api_key(
            config,
            prefix="judge_client",
            default_timeout=60.0,
        ),
        temperature=accessor.get_float("temperature", 0.0),
    )
