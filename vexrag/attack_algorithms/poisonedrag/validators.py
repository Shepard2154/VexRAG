import random
import re
from typing import Any

from vexrag.core.llm_response_validation import (
    LLMPayloadValidationError,
)
from vexrag.core.llm_response_validation import (
    coerce_payload_to_dict as _coerce_payload_to_dict,
)
from vexrag.core.llm_response_validation import (
    validate_correct_answer_payload as _core_validate_correct_answer,
)


class PoisonedRAGValidationError(LLMPayloadValidationError):
    """Raised when an LLM payload does not match expected schema."""


def coerce_payload_to_dict(payload: Any) -> dict[str, Any]:
    """Convert payload to dictionary, accepting dict or JSON string."""
    try:
        return _coerce_payload_to_dict(payload)
    except LLMPayloadValidationError as exc:
        raise PoisonedRAGValidationError(str(exc)) from exc


def validate_correct_answer_payload(payload: Any) -> str:
    """Extract stage-1 correct answer from payload."""
    try:
        return _core_validate_correct_answer(payload)
    except LLMPayloadValidationError as exc:
        raise PoisonedRAGValidationError(str(exc)) from exc


def validate_poison_payload(payload: Any) -> tuple[str, list[str]]:
    """Extract stage-2 incorrect answer and adversarial texts from payload."""
    data = coerce_payload_to_dict(payload)

    incorrect_answer = _first_present_string(
        data,
        (
            "incorrect_answer",
            "incorrect answer",
            "incorrectAnswer",
        ),
    )
    if not isinstance(incorrect_answer, str) or not incorrect_answer.strip():
        raise PoisonedRAGValidationError(
            "Field 'incorrect_answer' must be non-empty string."
        )

    adv_texts = data.get("adv_texts")
    if not isinstance(adv_texts, list):
        adv_texts = _collect_corpus_fields(data)
    if not isinstance(adv_texts, list):
        raise PoisonedRAGValidationError(
            "Field 'adv_texts' must be a list of strings "
            "or provide corpus1..corpusN fields."
        )

    cleaned_adv = [
        item.strip() for item in adv_texts if isinstance(item, str) and item.strip()
    ]
    if not cleaned_adv:
        raise PoisonedRAGValidationError(
            "Field 'adv_texts' must contain non-empty strings."
        )
    return incorrect_answer.strip(), cleaned_adv


def _collect_corpus_fields(data: dict[str, Any]) -> list[str] | None:
    pairs: list[tuple[int, str]] = []
    for key, value in data.items():
        match = re.fullmatch(r"corpus(\d+)", str(key))
        if match is None:
            continue
        if isinstance(value, str) and value.strip():
            pairs.append((int(match.group(1)), value.strip()))
    if not pairs:
        return None
    return [value for _index, value in sorted(pairs, key=lambda item: item[0])]


def _first_present_string(
    data: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_adv_texts(
    adv_texts: list[str], limit: int, seed: int | None = None
) -> list[str]:
    """Deduplicate and truncate adversarial texts with deterministic ordering by seed."""
    deduped = list(dict.fromkeys(adv_texts))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(deduped)

    return deduped[:limit]
