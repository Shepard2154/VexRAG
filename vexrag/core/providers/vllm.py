from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.core.providers.common import coerce_embedding, post_json
from vexrag.core.providers.config_accessor import ProviderConfigAccessor
from vexrag.core.providers.errors import ProviderConfigError, ProviderServiceError

_DEFAULT_API_KEY = "vllm-local"


def _normalize_api_key(api_key: str | None) -> str:
    if api_key is None:
        return _DEFAULT_API_KEY
    stripped = api_key.strip()
    return stripped or _DEFAULT_API_KEY


@dataclass(frozen=True, slots=True)
class VLLMEmbeddingClientConfig:
    model: str
    base_url: str
    endpoint: str
    api_key: str = _DEFAULT_API_KEY
    timeout: float | None = 30.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("embedding_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("embedding_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("embedding_client.endpoint is required")
        object.__setattr__(self, "api_key", _normalize_api_key(self.api_key))
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("embedding_client.timeout must be greater than 0")


class VLLMEmbeddingClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        api_key: str = _DEFAULT_API_KEY,
        timeout: float | None = 30.0,
    ) -> None:
        self._config = VLLMEmbeddingClientConfig(
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
        )

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


@dataclass(frozen=True, slots=True)
class VLLMJudgeClientConfig:
    model: str
    base_url: str
    endpoint: str
    api_key: str = _DEFAULT_API_KEY
    timeout: float | None = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("judge_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("judge_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("judge_client.endpoint is required")
        object.__setattr__(self, "api_key", _normalize_api_key(self.api_key))
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("judge_client.timeout must be greater than 0")


class VLLMJudgeClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        api_key: str = _DEFAULT_API_KEY,
        timeout: float | None = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self._config = VLLMJudgeClientConfig(
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
        )

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


def build_embedding_client(config: Mapping[str, Any]) -> VLLMEmbeddingClient:
    config_accessor = ProviderConfigAccessor(config, prefix="embedding_client")
    return VLLMEmbeddingClient(
        model=config_accessor.get_required_string("model"),
        base_url=config_accessor.get_required_string("base_url").rstrip("/"),
        endpoint=config_accessor.get_required_string("endpoint"),
        api_key=_normalize_api_key(config_accessor.get_optional_string("api_key")),
        timeout=config_accessor.get_optional_float("timeout", 30.0),
    )


def build_judge_client(config: Mapping[str, Any]) -> VLLMJudgeClient:
    config_accessor = ProviderConfigAccessor(config, prefix="judge_client")
    return VLLMJudgeClient(
        model=config_accessor.get_required_string("model"),
        base_url=config_accessor.get_required_string("base_url").rstrip("/"),
        endpoint=config_accessor.get_required_string("endpoint"),
        api_key=_normalize_api_key(config_accessor.get_optional_string("api_key")),
        timeout=config_accessor.get_optional_float("timeout", 60.0),
        temperature=config_accessor.get_float("temperature", 0.0),
    )
