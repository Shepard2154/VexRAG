import pytest

from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.evaluation import (
    CompositeEvaluator,
    EmbeddingSimilarityEvaluator,
    EvaluationDependencyError,
    LLMJudgeEvaluator,
    ProviderBackedEmbeddingClient,
    ProviderBackedJsonCompletionClient,
)
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.builder import build_evaluator
from vexrag.core.scan.config.errors import EvaluationConfigError


class _FailingEmbeddingClient:
    def embed_texts(self, texts):
        raise ProviderServiceError("embedding down")


class _FailingJudgeClient:
    def complete_json(self, prompt):
        raise ProviderServiceError("judge down")


class TestBuildEvaluator:
    def test_build_evaluator_rejects_legacy_evaluations_key(self) -> None:
        registries = create_scan_registries()
        cfg = {
            "evaluations": {
                "combine": "any",
                "evaluators": [{"strategy": "embedding_similarity"}],
            },
        }
        with pytest.raises(EvaluationConfigError, match="evaluations"):
            build_evaluator(cfg, attack_id="hijackrag", registries=registries)

    def test_build_evaluator_composite_bundle(self) -> None:
        registries = create_scan_registries()
        emb = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "endpoint": "/api/embed",
            "model": "m",
        }
        sem = {
            "metric": "cosine",
            "attack_similarity_threshold": 0.9,
            "max_reference_similarity": 0.5,
            "attack_margin_threshold": 0.05,
        }
        cfg = {
            "evaluation": {
                "strategy": "composite",
                "combine": "all",
                "evaluators": [
                    {
                        "strategy": "embedding_similarity",
                        "embedding_client": emb,
                        "embedding_similarity": sem,
                    },
                    {
                        "strategy": "embedding_similarity",
                        "embedding_client": emb,
                        "embedding_similarity": {
                            **sem,
                            "attack_similarity_threshold": 0.7,
                        },
                    },
                ],
            },
        }
        strat = build_evaluator(cfg, attack_id="hijackrag", registries=registries)
        assert isinstance(strat, CompositeEvaluator)

    def test_build_evaluator_rejects_unknown_similarity_metric(self) -> None:
        registries = create_scan_registries()
        emb = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "endpoint": "/api/embed",
            "model": "m",
        }
        cfg = {
            "evaluation": {
                "strategy": "embedding_similarity",
                "embedding_client": emb,
                "embedding_similarity": {
                    "metric": "euclidean",
                },
            },
        }
        with pytest.raises(EvaluationConfigError, match="metric must be one of"):
            build_evaluator(cfg, attack_id="hijackrag", registries=registries)

    def test_build_evaluator_wraps_provider_clients(self) -> None:
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
        assert isinstance(
            embedding_eval.embedding_client, ProviderBackedEmbeddingClient
        )

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


class TestProviderClientAdapters:
    def test_provider_backed_embedding_client_maps_provider_error(self) -> None:
        client = ProviderBackedEmbeddingClient(_FailingEmbeddingClient())
        with pytest.raises(
            EvaluationDependencyError, match="embedding down"
        ) as exc_info:
            client.embed_texts(["a"])
        assert isinstance(exc_info.value.__cause__, ProviderServiceError)

    def test_provider_backed_judge_client_maps_provider_error(self) -> None:
        client = ProviderBackedJsonCompletionClient(_FailingJudgeClient())
        with pytest.raises(EvaluationDependencyError, match="judge down") as exc_info:
            client.complete_json("prompt")
        assert isinstance(exc_info.value.__cause__, ProviderServiceError)
