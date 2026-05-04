import pytest

from vexrag.core.attack_plan import (
    AttackStepSpec,
    materialize_config_for_attack_id,
    materialize_step_config,
    parse_attack_steps,
    resolve_generate_cases_attack_id,
)
from vexrag.core.attacks.plugin import AttackPlugin
from vexrag.core.attacks.registry import AttackRegistry, AttackRegistryError
from vexrag.core.config_errors import ScanConfigError


def _stub_plugin(aid: str) -> AttackPlugin:
    return AttackPlugin(
        attack_id=aid,
        display_name=aid,
        build_scan_command=lambda c, b=None: object(),
        judge_prompt_builder_factory=None,
        build_automatic_case_generator=lambda c: object(),
        serialize_case_for_yaml=lambda x: {},
        generate_cases=lambda c, p: [],
    )


def test_parse_single_step() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    cfg = {
        "attacks": [{"id": "alpha", "params": {"x": 1}}],
        "scan": {"repetitions": 2},
        "evaluation": {"strategy": "semantic_similarity"},
    }
    steps = parse_attack_steps(cfg, reg)
    assert len(steps) == 1
    assert steps[0].attack_id == "alpha"
    assert steps[0].params == {"x": 1}


def test_parse_rejects_legacy_attack_key() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    with pytest.raises(ScanConfigError, match="legacy"):
        parse_attack_steps(
            {"attack": {"alpha": {}}, "attacks": [{"id": "alpha", "params": {}}]},
            reg,
        )


def test_materialize_merges_scan_and_evaluation() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    step = AttackStepSpec(
        attack_id="alpha",
        params={"k": 1},
        scan_override={"corpus_poisoning": {"filename_prefix": "pfx"}},
        evaluation_override={"strategy": "llm_judge"},
        evaluations_override=None,
    )
    root = {
        "target_system": {"http": {"base_url": "http://x"}},
        "scan": {
            "repetitions": 1,
            "corpus_poisoning": {"backend": "file_text", "path": "/tmp"},
        },
        "evaluation": {
            "strategy": "semantic_similarity",
            "semantic_similarity": {"metric": "cosine"},
        },
    }
    m = materialize_step_config(root, step)
    assert m["attack"] == {"alpha": {"k": 1}}
    assert m["scan"]["repetitions"] == 1
    assert m["scan"]["corpus_poisoning"]["backend"] == "file_text"
    assert m["scan"]["corpus_poisoning"]["filename_prefix"] == "pfx"
    assert m["evaluation"]["strategy"] == "llm_judge"
    assert m["evaluation"]["semantic_similarity"]["metric"] == "cosine"


def test_resolve_generate_cases_auto_single() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    cfg = {"attacks": [{"id": "alpha", "params": {}}]}
    assert resolve_generate_cases_attack_id(cfg, reg, explicit=None) == "alpha"


def test_resolve_generate_cases_auto_multi_requires_explicit() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    reg.register(_stub_plugin("beta"))
    cfg = {
        "attacks": [
            {"id": "alpha", "params": {}},
            {"id": "beta", "params": {}},
        ]
    }
    with pytest.raises(AttackRegistryError, match="multiple"):
        resolve_generate_cases_attack_id(cfg, reg, explicit=None)


def test_materialize_config_for_attack_id_picks_step() -> None:
    reg = AttackRegistry()
    reg.register(_stub_plugin("alpha"))
    reg.register(_stub_plugin("beta"))
    cfg = {
        "attacks": [
            {"id": "alpha", "params": {"a": 1}},
            {"id": "beta", "params": {"b": 2}},
        ]
    }
    m = materialize_config_for_attack_id(cfg, "beta", registry=reg)
    assert m["attack"] == {"beta": {"b": 2}}
