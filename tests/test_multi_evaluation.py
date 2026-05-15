import pytest

from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import EvaluationConfigError
from vexrag.core.config.build import build_evaluation_strategy
from vexrag.core.evaluation.multi_evaluator import MultiEvaluator
from vexrag.core.evaluation.protocols import EvaluationInput, EvaluationResult


class _FixedEvaluator:
    __slots__ = ("_ok", "_name")

    def __init__(self, name: str, ok: bool) -> None:
        self._name = name
        self._ok = ok

    @property
    def strategy(self) -> str:
        return self._name

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=self._ok,
            strategy=self._name,
            scores={"x": 1.0 if self._ok else 0.0},
            reason=f"reason-{self._name}",
        )


def test_multi_sub_evaluators_exposes_children() -> None:
    inner = (_FixedEvaluator("a", False), _FixedEvaluator("b", True))
    m = MultiEvaluator(inner, combine="any")
    assert m.sub_evaluators == inner


def test_multi_any_true_if_one_succeeds() -> None:
    m = MultiEvaluator(
        (_FixedEvaluator("a", False), _FixedEvaluator("b", True)),
        combine="any",
    )
    r = m.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.attack_successful is True
    assert "a" in r.strategy and "b" in r.strategy


def test_multi_all_false_if_one_fails() -> None:
    m = MultiEvaluator(
        (_FixedEvaluator("a", True), _FixedEvaluator("b", False)),
        combine="all",
    )
    r = m.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.attack_successful is False


def test_multi_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiEvaluator((), combine="any")


def test_build_evaluation_strategy_rejects_both_evaluation_keys() -> None:
    reg = AttackRegistry()
    reg.register(HIJACK_PLUGIN)
    reg.register(POISON_PLUGIN)
    cfg = {
        "evaluation": {"strategy": "embedding_similarity"},
        "evaluations": {
            "combine": "any",
            "evaluators": [{"strategy": "embedding_similarity"}],
        },
    }
    with pytest.raises(EvaluationConfigError, match="not both"):
        build_evaluation_strategy(cfg, attack_id="hijackrag", registry=reg)


def test_build_evaluation_strategy_evaluations_bundle() -> None:
    reg = AttackRegistry()
    reg.register(HIJACK_PLUGIN)
    reg.register(POISON_PLUGIN)
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
        "evaluations": {
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
    strat = build_evaluation_strategy(cfg, attack_id="hijackrag", registry=reg)
    assert isinstance(strat, MultiEvaluator)


def test_build_evaluation_strategy_accepts_legacy_semantic_similarity_alias() -> None:
    reg = AttackRegistry()
    reg.register(HIJACK_PLUGIN)
    reg.register(POISON_PLUGIN)
    emb = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "endpoint": "/api/embed",
        "model": "m",
    }
    cfg = {
        "evaluation": {
            "strategy": "semantic_similarity",
            "embedding_client": emb,
            "semantic_similarity": {
                "metric": "cosine",
                "attack_similarity_threshold": 0.9,
                "max_reference_similarity": 0.5,
                "attack_margin_threshold": 0.05,
            },
        },
    }
    from vexrag.core.evaluation.embedding_similarity_evaluator import (
        EmbeddingSimilarityEvaluator,
    )

    strat = build_evaluation_strategy(cfg, attack_id="hijackrag", registry=reg)
    assert isinstance(strat, EmbeddingSimilarityEvaluator)
    assert strat.attack_similarity_threshold == 0.9
