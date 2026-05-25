from dataclasses import dataclass, field

from vexrag.core.attack_configurator import (
    CorrectAnswerSource,
    PoisonedResult,
    TargetStyle,
)


@dataclass(slots=True)
class HijackRAGRequest:
    """Input for building HijackRAG-style poison documents from segment templates."""

    query: str
    hijack_insert: str
    correct_answer: str
    adv_per_query: int = 1
    case_id: str | None = None
    segment_ids: tuple[str, ...] = ()
    correct_answer_style: TargetStyle = "short_fact"
    seed: int | None = None


@dataclass(slots=True)
class HijackRAGMeta:
    latency_ms: int
    prompt_version: str
    model_id: str
    correct_answer_source: CorrectAnswerSource
    segment_ids: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HijackRAGResult(PoisonedResult):
    """Generated hijack documents and evaluation anchors."""

    query: str
    correct_answer: str
    incorrect_answer: str  # For HijackRAG this mirrors request.hijack_insert.
    adv_texts: list[str]
    meta: HijackRAGMeta
