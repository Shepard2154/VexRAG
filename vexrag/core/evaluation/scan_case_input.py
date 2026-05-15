from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    """Attack-agnostic input for evaluating a target-system answer.

    Required string fields must be non-empty after stripping (see ``__post_init__``).

    Attributes:
        query: User question sent to the target system.
        actual_answer: Answer produced by the target system under test.
        expected_clean_answer: Reference answer without a successful attack
            (benign / ground-truth baseline).
        expected_attack_answer: Reference answer when the attack succeeds
            (adversarial target outcome).
        retrieved_contexts: Context chunks the target system used or returned
            (e.g. RAG retrieval from ``TargetSystemResponse.contexts``).
        context_override: Optional extra context lines not in retrieval
            (e.g. adversarial probe texts when ``override_contexts`` is enabled).
            ``None`` means no override; use ``()`` for an explicit empty override.
        metadata: Opaque key-value data for evaluators and reports
            (e.g. ``poisoned_document_ids`` from ``probe_with_poisoning_and_evaluation``).
    """

    query: str
    actual_answer: str
    expected_clean_answer: str
    expected_attack_answer: str
    retrieved_contexts: tuple[str, ...] = ()
    context_override: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "query",
            "actual_answer",
            "expected_clean_answer",
            "expected_attack_answer",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def context_lines(self) -> tuple[str, ...]:
        lines: list[str] = list(self.retrieved_contexts)
        if self.context_override is not None:
            lines.extend(self.context_override)
        return tuple(lines)
