import pytest

from vexrag.core.evaluation import (
    EmbeddingSimilarityEvaluator,
    EvaluationDependencyError,
    EvaluationInput,
    EvaluationStrategy,
    EvaluatorError,
)


class _FailingEmbeddingClient:
    __slots__ = ("_exc",)

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def embed_texts(self, texts):
        raise self._exc


def _sample_input() -> EvaluationInput:
    return EvaluationInput(
        query="q",
        actual_answer="a",
        expected_clean_answer="c",
        expected_attack_answer="i",
    )


class TestEmbeddingSimilarityEvaluator:
    def test_embedding_similarity_maps_dependency_error_to_incomplete_result(
        self,
    ) -> None:
        ev = EmbeddingSimilarityEvaluator(
            embedding_client=_FailingEmbeddingClient(
                EvaluationDependencyError("svc down")
            )
        )
        r = ev.evaluate(_sample_input())
        assert r.completed is False
        assert r.attack_successful is False
        assert r.strategy == EvaluationStrategy.EMBEDDING_SIMILARITY
        assert "svc down" in (r.reason or "")

    def test_embedding_similarity_maps_wrong_embedding_count_to_incomplete_result(
        self,
    ) -> None:
        class _WrongCountClient:
            def embed_texts(self, texts):
                return ([1.0, 0.0], [0.0, 1.0])

        ev = EmbeddingSimilarityEvaluator(embedding_client=_WrongCountClient())
        r = ev.evaluate(_sample_input())
        assert r.completed is False
        assert r.attack_successful is False
        assert "one vector per input text" in (r.reason or "")

    def test_embedding_similarity_maps_evaluator_errors_to_incomplete_result(
        self,
    ) -> None:
        ev = EmbeddingSimilarityEvaluator(
            embedding_client=_FailingEmbeddingClient(EvaluatorError("boom"))
        )
        r = ev.evaluate(_sample_input())
        assert r.completed is False
        assert r.attack_successful is False
        assert "boom" in (r.reason or "")

    def test_embedding_similarity_maps_metric_errors_to_incomplete_result(self) -> None:
        class _ZeroNormClient:
            def embed_texts(self, texts):
                return ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])

        ev = EmbeddingSimilarityEvaluator(embedding_client=_ZeroNormClient())
        r = ev.evaluate(_sample_input())
        assert r.completed is False
        assert r.attack_successful is False
        assert "zero magnitude" in (r.reason or "")

    def test_embedding_similarity_propagates_unexpected_client_errors(self) -> None:
        ev = EmbeddingSimilarityEvaluator(
            embedding_client=_FailingEmbeddingClient(RuntimeError("not mapped"))
        )
        with pytest.raises(RuntimeError, match="not mapped"):
            ev.evaluate(_sample_input())
