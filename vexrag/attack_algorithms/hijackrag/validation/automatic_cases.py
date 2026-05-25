import re
from typing import Any

from vexrag.attack_algorithms.poison_base.case_validation import (
    normalize_case_id,
    required_text,
)
from vexrag.core.llm import LLMPayloadValidationError, coerce_payload_to_dict

_HIJACK_INSERT_RE = re.compile(r"^[A-Z][A-Z0-9_]{4,63}$")


class AutomaticCaseGenerationError(ValueError):
    """Raised when automatic case generation fails or returns invalid payload."""


def validate_cases_payload(
    payload: Any,
    *,
    expected_count: int,
) -> list[dict[str, str]]:
    try:
        data = coerce_payload_to_dict(payload)
    except LLMPayloadValidationError as exc:
        raise AutomaticCaseGenerationError(str(exc)) from exc
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise AutomaticCaseGenerationError("Field 'cases' must be a list.")
    if len(raw_cases) < expected_count:
        raise AutomaticCaseGenerationError(
            f"Expected at least {expected_count} generated cases, got {len(raw_cases)}."
        )

    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_markers: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        candidate = _validate_case_record(raw_case)
        if candidate["id"] in seen_ids or candidate["hijack_insert"] in seen_markers:
            continue
        seen_ids.add(candidate["id"])
        seen_markers.add(candidate["hijack_insert"])
        validated.append(candidate)
        if len(validated) >= expected_count:
            break

    if len(validated) < expected_count:
        raise AutomaticCaseGenerationError(
            f"Could not validate {expected_count} unique cases from model output."
        )
    return validated


def _validate_case_record(raw_case: dict[str, object]) -> dict[str, str]:
    query = required_text(
        raw_case.get("query"),
        field="query",
        error_type=AutomaticCaseGenerationError,
    )
    correct_answer = required_text(
        raw_case.get("correct_answer"),
        field="correct_answer",
        error_type=AutomaticCaseGenerationError,
    )
    hijack_insert = _required_hijack_insert(raw_case.get("hijack_insert"))
    raw_id = raw_case.get("id")
    case_id = normalize_case_id(raw_id if isinstance(raw_id, str) else query)
    return {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "hijack_insert": hijack_insert,
    }


def _required_hijack_insert(value: object) -> str:
    text = required_text(
        value,
        field="hijack_insert",
        error_type=AutomaticCaseGenerationError,
    )
    if not _HIJACK_INSERT_RE.match(text):
        raise AutomaticCaseGenerationError(
            "Field 'hijack_insert' must match SCREAMING_SNAKE_CASE (A-Z then A-Z0-9_), "
            f"length 5-64; got {text!r}"
        )
    return text
