from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vexrag.core.attacks.command import ScanCommandProtocol
from vexrag.core.evaluation import JudgePromptBuilderProtocol


@dataclass(frozen=True, slots=True)
class GenerateCasesParams:
    """Arguments shared by ``vx generate-cases`` across attacks."""

    count: int
    topic: str | None
    target_style: str
    seed: int | None
    adv_per_query: int = 1


@dataclass(frozen=True, slots=True)
class AttackPlugin:
    """One attack implementation registered under ``attack.<attack_id>`` in YAML."""

    attack_id: str
    display_name: str
    build_scan_command: Callable[[Mapping[str, Any], Path | None], ScanCommandProtocol]
    judge_prompt_builder_factory: Callable[[], JudgePromptBuilderProtocol] | None
    build_automatic_case_generator: Callable[[Mapping[str, Any]], Any]
    serialize_case_for_yaml: Callable[[Any], Mapping[str, Any]]
    generate_cases: Callable[
        [Mapping[str, Any], GenerateCasesParams],
        list[Any],
    ]
