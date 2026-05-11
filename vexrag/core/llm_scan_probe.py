"""Pre-scan connectivity checks for LLM-backed scan flows."""

from collections.abc import Mapping
from typing import Any

from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config.build import (
    attack_llm_client_section,
    attack_section,
    build_evaluation_strategy,
)
from vexrag.core.evaluation.llm_judge import LLMJudgeEvaluator
from vexrag.core.evaluation.multi import MultiEvaluator
from vexrag.core.evaluation.protocols import (
    EvaluationStrategyProtocol,
    JudgeLLMProtocol,
)
from vexrag.core.providers import build_judge_client as build_provider_judge_client
from vexrag.core.providers.errors import ProviderServiceError

_SCAN_LLM_PROBE_PROMPT = (
    'Return only a JSON object, no other text: {"vexrag_probe":"ok"}'
)


def _probe_complete_json(client: JudgeLLMProtocol, *, role: str) -> None:
    try:
        client.complete_json(_SCAN_LLM_PROBE_PROMPT)
    except Exception as exc:
        raise ProviderServiceError(f"LLM unavailable for scan ({role}): {exc}") from exc


def _probe_llm_judge_evaluators(
    strategy: EvaluationStrategyProtocol,
    *,
    prefix: str,
) -> None:
    if isinstance(strategy, LLMJudgeEvaluator):
        _probe_complete_json(
            strategy.judge_client,
            role=f"{prefix} — judge",
        )
        return
    if isinstance(strategy, MultiEvaluator):
        for sub in strategy.sub_evaluators:
            _probe_llm_judge_evaluators(sub, prefix=prefix)


def probe_scan_llms_for_materialized_config(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    step_label: str | None = None,
) -> None:
    """Raise ``ProviderServiceError`` if attack or judge LLM cannot serve a probe call."""
    label = step_label if step_label is not None else f"attack step ({attack_id})"
    attack_conf = attack_section(config, attack_id)
    llm_cfg = attack_llm_client_section(config, attack_conf, attack=attack_id)
    attack_client = build_provider_judge_client(llm_cfg)
    _probe_complete_json(attack_client, role=f"{label} — generator")

    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)

    evaluation = build_evaluation_strategy(
        config,
        attack_id=attack_id,
        registry=registry,
    )
    _probe_llm_judge_evaluators(evaluation, prefix=label)
