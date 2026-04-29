from collections.abc import Mapping
from typing import Any

from vexrag.core.evaluation import EmbeddingClientProtocol, JudgeLLMProtocol
from vexrag.core.providers.errors import ProviderConfigError, ProviderServiceError
from vexrag.core.providers.ollama import (
    OllamaEmbeddingClient,
    OllamaJudgeClient,
)
from vexrag.core.providers.ollama import (
    build_embedding_client as build_ollama_embedding_client,
)
from vexrag.core.providers.ollama import (
    build_judge_client as build_ollama_judge_client,
)


def build_embedding_client(config: Mapping[str, Any]) -> EmbeddingClientProtocol:
    provider = _provider_name(config, "embedding_client")
    if provider == "ollama":
        return build_ollama_embedding_client(config)
    raise ProviderConfigError(f"embedding_client.provider is not supported: {provider}")


def build_judge_client(config: Mapping[str, Any]) -> JudgeLLMProtocol:
    provider = _provider_name(config, "judge_client")
    if provider == "ollama":
        return build_ollama_judge_client(config)
    raise ProviderConfigError(f"judge_client.provider is not supported: {provider}")


def _provider_name(config: Mapping[str, Any], prefix: str) -> str:
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ProviderConfigError(f"{prefix}.provider is required")
    return provider.strip()


__all__ = [
    "OllamaEmbeddingClient",
    "OllamaJudgeClient",
    "ProviderConfigError",
    "ProviderServiceError",
    "build_embedding_client",
    "build_judge_client",
]
