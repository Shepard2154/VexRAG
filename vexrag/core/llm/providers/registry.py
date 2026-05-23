from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vexrag.core.llm.contracts import EmbeddingClient, JsonCompletionClient
from vexrag.core.llm.providers.errors import ProviderConfigError

EmbeddingClientBuilder = Callable[[Mapping[str, Any]], EmbeddingClient]
JsonCompletionClientBuilder = Callable[[Mapping[str, Any]], JsonCompletionClient]


@dataclass(frozen=True, slots=True)
class LLMProviderRegistry:
    embedding_builders: Mapping[str, EmbeddingClientBuilder]
    json_completion_builders: Mapping[str, JsonCompletionClientBuilder]

    def build_embedding_client(self, config: Mapping[str, Any]) -> EmbeddingClient:
        provider = _provider_name(config, "embedding_client")
        try:
            builder = self.embedding_builders[provider]
        except KeyError as err:
            raise ProviderConfigError(
                f"embedding_client.provider is not supported: {provider}"
            ) from err
        return builder(config)

    def build_json_completion_client(
        self,
        config: Mapping[str, Any],
    ) -> JsonCompletionClient:
        provider = _provider_name(config, "judge_client")
        try:
            builder = self.json_completion_builders[provider]
        except KeyError as err:
            raise ProviderConfigError(
                f"judge_client.provider is not supported: {provider}"
            ) from err
        return builder(config)


def _provider_name(config: Mapping[str, Any], prefix: str) -> str:
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ProviderConfigError(f"{prefix}.provider is required")
    return provider.strip()
