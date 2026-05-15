import pytest

from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import EvaluationConfigError
from vexrag.core.config.build import build_evaluator
from vexrag.core.evaluation.attack_verdict import CombineMode, EvaluationResult
from vexrag.core.evaluation.composite_evaluator import CompositeEvaluator
from vexrag.core.evaluation.scan_case_input import EvaluationInput


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


def test_composite_sub_evaluators_exposes_children() -> None:
    inner = (_FixedEvaluator("a", False), _FixedEvaluator("b", True))
    composite = CompositeEvaluator(inner, combine=CombineMode.ANY)
    assert composite.sub_evaluators == inner


def test_composite_any_true_if_one_succeeds() -> None:
    composite = CompositeEvaluator(
        (_FixedEvaluator("a", False), _FixedEvaluator("b", True)),
        combine=CombineMode.ANY,
    )
    r = composite.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.attack_successful is True
    assert "a" in r.strategy and "b" in r.strategy


def test_composite_all_false_if_one_fails() -> None:
    composite = CompositeEvaluator(
        (_FixedEvaluator("a", True), _FixedEvaluator("b", False)),
        combine=CombineMode.ALL,
    )
    r = composite.evaluate(
        EvaluationInput(
            query="q",
            actual_answer="x",
            expected_clean_answer="c",
            expected_attack_answer="i",
        )
    )
    assert r.attack_successful is False


def test_composite_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CompositeEvaluator((), combine=CombineMode.ANY)


def test_build_evaluator_rejects_legacy_evaluations_key() -> None:
    reg = AttackRegistry()
    reg.register(HIJACK_PLUGIN)
    reg.register(POISON_PLUGIN)
    cfg = {
        "evaluations": {
            "combine": "any",
            "evaluators": [{"strategy": "embedding_similarity"}],
        },
    }
    with pytest.raises(EvaluationConfigError, match="evaluations"):
        build_evaluator(cfg, attack_id="hijackrag", registry=reg)


def test_build_evaluator_composite_bundle() -> None:
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
    strat = build_evaluator(cfg, attack_id="hijackrag", registry=reg)
    assert isinstance(strat, CompositeEvaluator)


def test_build_evaluator_rejects_unknown_similarity_metric() -> None:
    reg = AttackRegistry()
    reg.register(HIJACK_PLUGIN)
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
        build_evaluator(cfg, attack_id="hijackrag", registry=reg)
