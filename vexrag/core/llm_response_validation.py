import json
from typing import Any


class LLMPayloadValidationError(ValueError):
    """Raised when an LLM JSON payload is missing or malformed."""


def coerce_payload_to_dict(payload: Any) -> dict[str, Any]:
    """Convert payload to dictionary, accepting dict or JSON string."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMPayloadValidationError("LLM response is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMPayloadValidationError("LLM response must be a JSON object.")
        return parsed
    raise LLMPayloadValidationError("Unsupported LLM response payload type.")


def _first_present_string(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def validate_correct_answer_payload(payload: Any) -> str:
    """Extract stage-1 correct answer from payload."""
    data = coerce_payload_to_dict(payload)
    correct_answer = _first_present_string(
        data,
        (
            "correct_answer",
            "correct answer",
            "correctAnswer",
        ),
    )
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        raise LLMPayloadValidationError(
            "Field 'correct_answer' must be non-empty string."
        )
    return correct_answer.strip()
