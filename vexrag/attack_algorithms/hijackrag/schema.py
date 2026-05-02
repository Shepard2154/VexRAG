from dataclasses import dataclass, field
from typing import Literal

from vexrag.attack_algorithms.poisonedrag.schema import TargetStyle

CorrectAnswerSource = Literal["provided", "target_system", "llm_generated"]


@dataclass(slots=True)
class HijackRAGRequest:
    """Input for building HijackRAG-style poison documents from segment templates."""

    query: str
    hijack_insert: str
    case_id: str | None = None
    correct_answer: str | None = None
    adv_per_query: int = 3
    segment_ids: tuple[str, ...] = ()
    target_style: TargetStyle = "short_fact"
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
class HijackRAGResult:
    """Generated hijack documents and evaluation anchors."""

    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]
    meta: HijackRAGMeta
