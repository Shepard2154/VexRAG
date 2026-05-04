import pytest

from vexrag.core.evaluation.multi import MultiEvaluator
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
    from vexrag.core.config_errors import EvaluationConfigError
    from vexrag.core.scan_config_build import build_evaluation_strategy

    from vexrag.core.attacks import default_attack_registry, ensure_builtin_attacks_registered

    ensure_builtin_attacks_registered()
    reg = default_attack_registry()
    cfg = {
        "evaluation": {"strategy": "semantic_similarity"},
        "evaluations": {"combine": "any", "evaluators": [{"strategy": "semantic_similarity"}]},
    }
    with pytest.raises(EvaluationConfigError, match="not both"):
        build_evaluation_strategy(cfg, attack_id="hijackrag", registry=reg)


def test_build_evaluation_strategy_evaluations_bundle() -> None:
    from vexrag.core.evaluation.multi import MultiEvaluator
    from vexrag.core.scan_config_build import build_evaluation_strategy

    from vexrag.core.attacks import default_attack_registry, ensure_builtin_attacks_registered

    ensure_builtin_attacks_registered()
    reg = default_attack_registry()
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
                {"strategy": "semantic_similarity", "embedding_client": emb, "semantic_similarity": sem},
                {
                    "strategy": "semantic_similarity",
                    "embedding_client": emb,
                    "semantic_similarity": {
                        **sem,
                        "attack_similarity_threshold": 0.7,
                    },
                },
            ],
        },
    }
    strat = build_evaluation_strategy(cfg, attack_id="hijackrag", registry=reg)
    assert isinstance(strat, MultiEvaluator)
