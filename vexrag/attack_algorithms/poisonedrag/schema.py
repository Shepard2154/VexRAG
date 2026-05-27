from dataclasses import dataclass, field
from typing import Literal

from vexrag.attack_algorithms.poisonedrag.scan_profile import POISONEDRAG_SCAN_PROFILE
from vexrag.core.attack_configurator import (
    CorrectAnswerSource,
    PoisonedResult,
    TargetStyle,
)

PoisoningStyle = Literal["original", "aggressive", "soft"]


@dataclass(slots=True)
class PoisonedRAGRequest:
    """Input contract for generating PoisonedRAG candidates."""

    query: str
    correct_answer: str | None = None
    target_incorrect_answer: str | None = None
    case_id: str | None = None
    adv_per_query: int = POISONEDRAG_SCAN_PROFILE.default_adv_per_query
    target_style: TargetStyle = "short_fact"
    poisoning_style: PoisoningStyle = "original"
    seed: int | None = None


@dataclass(slots=True)
class PoisonedRAGMeta:
    """Generation metadata for diagnostics and observability."""

    latency_ms: int
    prompt_version: str
    model_id: str
    correct_answer_source: CorrectAnswerSource
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PoisonedRAGResult(PoisonedResult):
    """Output contract returned by PoisonedRAG generation API."""

    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]
    meta: PoisonedRAGMeta
