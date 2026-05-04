"""Smoke: chain YAML builds a two-step command with MultiEvaluator on each step (no network)."""

from pathlib import Path

import pytest
import yaml

from vexrag.cli.scan_builder import build_scan_command
from vexrag.core.attacks import ensure_builtin_attacks_registered
from vexrag.core.attacks.chain_command import AttackChainScanCommand
from vexrag.core.attacks.command import ConfiguredScanCommand
from vexrag.core.evaluation.multi import MultiEvaluator


@pytest.fixture(scope="module", autouse=True)
def _register_builtin_attacks() -> None:
    ensure_builtin_attacks_registered()


def test_chain_example_yaml_wires_two_steps_and_multi_evaluator() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "RAG examples/small/rag_01_in_memory_en/scan_configs_examples"
        / "vexrag-chain-hijack-then-poisoned-semantic-ollama-nomic.yaml"
    )
    assert path.is_file(), f"missing example config: {path}"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    command = build_scan_command(config, base_dir=path.parent)

    assert isinstance(command, AttackChainScanCommand)
    steps = getattr(command, "_steps", ())
    assert len(steps) == 2
    assert steps[0][0] == "hijackrag"
    assert steps[1][0] == "poisonedrag"

    for attack_id, sub in steps:
        assert isinstance(sub, ConfiguredScanCommand)
        strat = sub.runner.evaluation_strategy
        assert isinstance(
            strat, MultiEvaluator
        ), f"{attack_id}: expected MultiEvaluator, got {type(strat)!r}"
