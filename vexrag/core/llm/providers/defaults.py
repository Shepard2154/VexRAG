from vexrag.core.llm.providers import ollama, sentence_transformers, vllm
from vexrag.core.llm.providers.registry import LLMProviderRegistry


def create_default_llm_provider_registry() -> LLMProviderRegistry:
    return LLMProviderRegistry(
        embedding_builders={
            "ollama": ollama.build_embedding_client,
            "sentence_transformers": sentence_transformers.build_embedding_client,
            "vllm": vllm.build_embedding_client,
        },
        json_completion_builders={
            "ollama": ollama.build_json_completion_client,
            "vllm": vllm.build_json_completion_client,
        },
    )
