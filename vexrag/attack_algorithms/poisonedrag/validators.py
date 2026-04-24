import json
import random
from typing import Any


class PoisonedRAGValidationError(ValueError):
    """Raised when an LLM payload does not match expected schema."""


def coerce_payload_to_dict(payload: Any) -> dict[str, Any]:
    """Convert payload to dictionary, accepting dict or JSON string."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PoisonedRAGValidationError("LLM response is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise PoisonedRAGValidationError("LLM response must be a JSON object.")
        return parsed
    raise PoisonedRAGValidationError("Unsupported LLM response payload type.")


def validate_correct_answer_payload(payload: Any) -> str:
    """Extract stage-1 correct answer from payload."""
    data = coerce_payload_to_dict(payload)
    correct_answer = data.get("correct_answer")
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        raise PoisonedRAGValidationError(
            "Field 'correct_answer' must be non-empty string."
        )
    return correct_answer.strip()


def validate_poison_payload(payload: Any) -> tuple[str, list[str]]:
    """Extract stage-2 incorrect answer and adversarial texts from payload."""
    data = coerce_payload_to_dict(payload)

    incorrect_answer = data.get("incorrect_answer")
    if not isinstance(incorrect_answer, str) or not incorrect_answer.strip():
        raise PoisonedRAGValidationError(
            "Field 'incorrect_answer' must be non-empty string."
        )

    adv_texts = data.get("adv_texts")
    if not isinstance(adv_texts, list):
        raise PoisonedRAGValidationError("Field 'adv_texts' must be a list of strings.")

    cleaned_adv = [
        item.strip() for item in adv_texts if isinstance(item, str) and item.strip()
    ]
    if not cleaned_adv:
        raise PoisonedRAGValidationError(
            "Field 'adv_texts' must contain non-empty strings."
        )
    return incorrect_answer.strip(), cleaned_adv


def normalize_adv_texts(
    adv_texts: list[str], limit: int, seed: int | None = None
) -> list[str]:
    """Deduplicate and truncate adversarial texts with deterministic ordering by seed."""
    deduped = list(dict.fromkeys(adv_texts))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(deduped)

    return deduped[:limit]
