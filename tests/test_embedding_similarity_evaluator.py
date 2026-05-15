import pytest

from vexrag.core.errors import ProviderServiceError
from vexrag.core.evaluation.embedding_similarity_evaluator import (
    EmbeddingSimilarityEvaluator,
)
from vexrag.core.evaluation.errors import EvaluatorError
from vexrag.core.evaluation.metrics.errors import ZeroNormEmbeddingError
from vexrag.core.evaluation.protocols import EvaluationInput
from vexrag.core.evaluation.strategies import EvaluationStrategy


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


def test_embedding_similarity_maps_provider_service_error_to_incomplete_result() -> (
    None
):
    ev = EmbeddingSimilarityEvaluator(
        embedding_client=_FailingEmbeddingClient(ProviderServiceError("svc down"))
    )
    r = ev.evaluate(_sample_input())
    assert r.evaluation_completed is False
    assert r.attack_successful is False
    assert r.strategy == EvaluationStrategy.EMBEDDING_SIMILARITY
    assert "svc down" in (r.reason or "")


def test_embedding_similarity_maps_wrong_embedding_count_to_incomplete_result() -> None:
    class _WrongCountClient:
        def embed_texts(self, texts):
            return ([1.0, 0.0], [0.0, 1.0])

    ev = EmbeddingSimilarityEvaluator(embedding_client=_WrongCountClient())
    r = ev.evaluate(_sample_input())
    assert r.evaluation_completed is False
    assert r.attack_successful is False
    assert "one vector per input text" in (r.reason or "")


def test_embedding_similarity_propagates_evaluator_errors() -> None:
    ev = EmbeddingSimilarityEvaluator(
        embedding_client=_FailingEmbeddingClient(EvaluatorError("boom"))
    )
    with pytest.raises(EvaluatorError, match="boom"):
        ev.evaluate(_sample_input())


def test_embedding_similarity_propagates_metric_errors() -> None:
    class _ZeroNormClient:
        def embed_texts(self, texts):
            return ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])

    ev = EmbeddingSimilarityEvaluator(embedding_client=_ZeroNormClient())
    with pytest.raises(ZeroNormEmbeddingError, match="zero magnitude"):
        ev.evaluate(_sample_input())


def test_embedding_similarity_propagates_unexpected_client_errors() -> None:
    ev = EmbeddingSimilarityEvaluator(
        embedding_client=_FailingEmbeddingClient(RuntimeError("not mapped"))
    )
    with pytest.raises(RuntimeError, match="not mapped"):
        ev.evaluate(_sample_input())
