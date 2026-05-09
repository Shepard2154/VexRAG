import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vexrag.core.providers.errors import ProviderConfigError, ProviderServiceError


@dataclass(frozen=True, slots=True)
class VLLMEmbeddingClientConfig:
    model: str
    base_url: str
    endpoint: str
    api_key: str
    timeout: float | None = 30.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("embedding_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("embedding_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("embedding_client.endpoint is required")
        if not self.api_key:
            raise ProviderConfigError("embedding_client.api_key is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("embedding_client.timeout must be greater than 0")


class VLLMEmbeddingClient:
    __slots__ = ("_config",)

    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        api_key: str = "ANYTHING",
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
        response = _post_vllm_json(
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
            vectors.append(_coerce_embedding(embedding))
        return tuple(vectors)


@dataclass(frozen=True, slots=True)
class VLLMJudgeClientConfig:
    model: str
    base_url: str
    endpoint: str
    api_key: str
    timeout: float | None = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ProviderConfigError("judge_client.model is required")
        if not self.base_url:
            raise ProviderConfigError("judge_client.base_url is required")
        if not self.endpoint:
            raise ProviderConfigError("judge_client.endpoint is required")
        if not self.api_key:
            raise ProviderConfigError("judge_client.api_key is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ProviderConfigError("judge_client.timeout must be greater than 0")


class VLLMJudgeClient:
    __slots__ = ("_config",)

    def __init__(
        self,
        model: str,
        base_url: str,
        endpoint: str,
        api_key: str = "ANYTHING",
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
        response = _post_vllm_json(
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
    return VLLMEmbeddingClient(
        model=_required_string(config, "model", "embedding_client"),
        base_url=_required_string(config, "base_url", "embedding_client").rstrip("/"),
        endpoint=_required_string(config, "endpoint", "embedding_client"),
        api_key=_optional_string_option(config, "api_key", "ANYTHING"),
        timeout=_optional_float_option(config, "timeout", 30.0),
    )


def build_judge_client(config: Mapping[str, Any]) -> VLLMJudgeClient:
    return VLLMJudgeClient(
        model=_required_string(config, "model", "judge_client"),
        base_url=_required_string(config, "base_url", "judge_client").rstrip("/"),
        endpoint=_required_string(config, "endpoint", "judge_client"),
        api_key=_optional_string_option(config, "api_key", "ANYTHING"),
        timeout=_optional_float_option(config, "timeout", 60.0),
        temperature=_float_option(config, "temperature", 0.0),
    )


def _float_option(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ProviderConfigError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigError(f"{key} must be a number") from exc


def _optional_float_option(
    config: Mapping[str, Any],
    key: str,
    default: float,
) -> float | None:
    value = config.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProviderConfigError(f"{key} must be a number or null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigError(f"{key} must be a number or null") from exc


def _required_string(config: Mapping[str, Any], key: str, prefix: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"{prefix}.{key} is required")
    return value.strip()


def _optional_string_option(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _post_vllm_json(
    *,
    base_url: str,
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float | None,
    service_name: str,
) -> Mapping[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ProviderServiceError(
            f"{service_name} request returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise ProviderServiceError(
            f"{service_name} request failed: {error.reason}"
        ) from error

    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderServiceError(
            f"{service_name} response was not valid JSON"
        ) from error
    if not isinstance(decoded, Mapping):
        raise ProviderServiceError(f"{service_name} response must be a JSON object")
    return decoded


def _coerce_embedding(vector: object) -> tuple[float, ...]:
    if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
        raise ProviderServiceError("embedding response items must be numeric lists")
    try:
        return tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ProviderServiceError("embedding values must be numeric") from exc
