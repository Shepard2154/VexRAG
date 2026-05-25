from typing import Any

from vexrag.attack_algorithms.poison_base.case_validation import (
    normalize_case_id,
    required_text,
)
from vexrag.core.llm import LLMPayloadValidationError, coerce_payload_to_dict


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

    by_case_id: dict[str, dict[str, str]] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
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
        target_incorrect_answer = required_text(
            raw_case.get("target_incorrect_answer"),
            field="target_incorrect_answer",
            error_type=AutomaticCaseGenerationError,
        )
        raw_id = raw_case.get("id")
        case_id = normalize_case_id(raw_id if isinstance(raw_id, str) else query)
        if case_id in by_case_id:
            continue
        by_case_id[case_id] = {
            "id": case_id,
            "query": query,
            "correct_answer": correct_answer,
            "target_incorrect_answer": target_incorrect_answer,
        }
        if len(by_case_id) >= expected_count:
            break

    validated = list(by_case_id.values())
    if len(validated) < expected_count:
        raise AutomaticCaseGenerationError(
            f"Could not validate {expected_count} unique cases from model output."
        )
    return validated
