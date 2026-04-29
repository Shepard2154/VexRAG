from dataclasses import dataclass, field
from typing import Literal

TargetStyle = Literal["short_fact", "paragraph"]
CorrectAnswerSource = Literal["provided", "target_system", "llm_generated"]


@dataclass(slots=True)
class PoisonedRAGRequest:
    """Input contract for generating PoisonedRAG candidates."""

    query: str
    case_id: str | None = None
    correct_answer: str | None = None
    target_incorrect_answer: str | None = None
    adv_per_query: int = 3
    target_style: TargetStyle = "short_fact"
    seed: int | None = None


@dataclass(slots=True)
class PoisonedRAGSample:
    """Single generated adversarial candidate."""

    text: str
    source_corpus: str | None = None


@dataclass(slots=True)
class PoisonedRAGMeta:
    """Generation metadata for diagnostics and observability."""

    latency_ms: int
    prompt_version: str
    model_id: str
    correct_answer_source: CorrectAnswerSource
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PoisonedRAGResult:
    """Output contract returned by PoisonedRAG generation API."""

    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]
    meta: PoisonedRAGMeta
