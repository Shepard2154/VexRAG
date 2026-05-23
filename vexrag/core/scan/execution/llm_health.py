from collections.abc import Mapping
from typing import Any

from vexrag.core.evaluation.contracts import Evaluator
from vexrag.core.llm.contracts import JsonCompletionClient
from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.builder import (
    attack_llm_client_section,
    attack_section,
    build_evaluator,
)
from vexrag.core.scan.builder.registries import ScanRegistries

_SCAN_LLM_PROBE_PROMPT = (
    'Return only a JSON object, no other text: {"vexrag_probe":"ok"}'
)


def probe_complete_json(client: JsonCompletionClient, *, role: str) -> None:
    try:
        client.complete_json(_SCAN_LLM_PROBE_PROMPT)
    except ProviderServiceError as exc:
        raise ProviderServiceError(f"LLM unavailable for scan ({role}): {exc}") from exc


def _probe_llm_judge_evaluators(
    strategy: Evaluator,
    *,
    prefix: str,
) -> None:
    client = getattr(strategy, "judge_client", None)
    if client is not None:
        probe_complete_json(client, role=f"{prefix} - judge")
    sub_evaluators = getattr(strategy, "sub_evaluators", None)
    if sub_evaluators is not None:
        for sub in sub_evaluators:
            _probe_llm_judge_evaluators(sub, prefix=prefix)


def probe_scan_llms_for_materialized_config(
    config: Mapping[str, Any],
    *,
    attack_id: str,
    registries: ScanRegistries,
    step_label: str | None = None,
) -> None:
    """Raise ProviderServiceError if attack or judge LLM cannot serve a probe call."""
    label = step_label if step_label is not None else f"attack step ({attack_id})"
    attack_conf = attack_section(config, attack_id)
    llm_cfg = attack_llm_client_section(config, attack_conf, attack=attack_id)
    attack_client = registries.llm_providers.build_json_completion_client(llm_cfg)
    probe_complete_json(attack_client, role=f"{label} - generator")

    evaluation = build_evaluator(
        config,
        attack_id=attack_id,
        registries=registries,
    )
    _probe_llm_judge_evaluators(evaluation, prefix=label)
