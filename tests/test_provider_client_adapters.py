import pytest

from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.evaluation import (
    EmbeddingSimilarityEvaluator,
    EvaluationDependencyError,
    LLMJudgeEvaluator,
    ProviderBackedEmbeddingClient,
    ProviderBackedJsonCompletionClient,
)
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.builder import build_evaluator


class _FailingEmbeddingClient:
    def embed_texts(self, texts):
        raise ProviderServiceError("embedding down")


class _FailingJudgeClient:
    def complete_json(self, prompt):
        raise ProviderServiceError("judge down")


def test_provider_backed_embedding_client_maps_provider_error() -> None:
    client = ProviderBackedEmbeddingClient(_FailingEmbeddingClient())
    with pytest.raises(EvaluationDependencyError, match="embedding down") as exc_info:
        client.embed_texts(["a"])
    assert isinstance(exc_info.value.__cause__, ProviderServiceError)


def test_provider_backed_judge_client_maps_provider_error() -> None:
    client = ProviderBackedJsonCompletionClient(_FailingJudgeClient())
    with pytest.raises(EvaluationDependencyError, match="judge down") as exc_info:
        client.complete_json("prompt")
    assert isinstance(exc_info.value.__cause__, ProviderServiceError)


def test_build_evaluator_wraps_provider_clients() -> None:
    registries = create_scan_registries()
    emb = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "endpoint": "/api/embed",
        "model": "m",
    }
    judge = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "endpoint": "/api/chat",
        "model": "m",
    }
    embedding_cfg = {
        "evaluation": {
            "strategy": "embedding_similarity",
            "embedding_client": emb,
            "embedding_similarity": {"metric": "cosine"},
        },
    }
    embedding_eval = build_evaluator(
        embedding_cfg,
        attack_id="hijackrag",
        registries=registries,
    )
    assert isinstance(embedding_eval, EmbeddingSimilarityEvaluator)
    assert isinstance(embedding_eval.embedding_client, ProviderBackedEmbeddingClient)

    judge_cfg = {
        "evaluation": {
            "strategy": "llm_judge",
            "judge_client": judge,
        },
    }
    judge_eval = build_evaluator(
        judge_cfg, attack_id="hijackrag", registries=registries
    )
    assert isinstance(judge_eval, LLMJudgeEvaluator)
    assert isinstance(judge_eval.judge_client, ProviderBackedJsonCompletionClient)
