from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.core.providers.common import coerce_embedding, post_json
from vexrag.core.providers.config_accessor import ProviderConfigAccessor
from vexrag.core.providers.errors import ProviderConfigError, ProviderServiceError


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingClientConfig:
    model: str
    base_url: str
    endpoint: str
    timeout: float | None = 30.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("embedding_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("embedding_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("embedding_client.endpoint is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("embedding_client.timeout must be greater than 0")


class OllamaEmbeddingClient:
    __slots__ = ("_config",)

    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        timeout: float | None = 30.0,
    ) -> None:
        config = OllamaEmbeddingClientConfig(
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            timeout=timeout,
        )
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


@dataclass(frozen=True, slots=True)
class OllamaJudgeClientConfig:
    model: str
    base_url: str
    endpoint: str
    timeout: float | None = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("judge_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("judge_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("judge_client.endpoint is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("judge_client.timeout must be greater than 0")


class OllamaJudgeClient:
    __slots__ = ("_config",)

    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        timeout: float | None = 60.0,
        temperature: float = 0.0,
    ) -> None:
        config = OllamaJudgeClientConfig(
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            timeout=timeout,
            temperature=temperature,
        )
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

    @property
    def temperature(self) -> float:
        return self._config.temperature

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
    config_accessor = ProviderConfigAccessor(config, prefix="embedding_client")
    return OllamaEmbeddingClient(
        model=config_accessor.get_required_string("model"),
        base_url=config_accessor.get_required_string("base_url").rstrip("/"),
        endpoint=config_accessor.get_required_string("endpoint"),
        timeout=config_accessor.get_optional_float("timeout", 30.0),
    )


def build_judge_client(config: Mapping[str, Any]) -> OllamaJudgeClient:
    config_accessor = ProviderConfigAccessor(config, prefix="judge_client")
    return OllamaJudgeClient(
        model=config_accessor.get_required_string("model"),
        base_url=config_accessor.get_required_string("base_url").rstrip("/"),
        endpoint=config_accessor.get_required_string("endpoint"),
        timeout=config_accessor.get_optional_float("timeout", 60.0),
        temperature=config_accessor.get_float("temperature", 0.0),
    )
