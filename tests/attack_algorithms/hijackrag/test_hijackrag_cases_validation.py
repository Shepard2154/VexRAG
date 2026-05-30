import pytest

from vexrag.attack_algorithms.hijackrag.validation import (
    AutomaticCaseGenerationError,
    validate_cases_payload,
)


def _valid_hijack_case(
    case_id: str,
    query: str = "What is RAG?",
    *,
    hijack_insert: str = "PWNED_BY_HIJACK",
) -> dict[str, str]:
    return {
        "id": case_id,
        "query": query,
        "correct_answer": "Clean answer.",
        "hijack_insert": hijack_insert,
    }


class TestHijackRagValidateCasesPayload:
    def test_validate_cases_payload_accepts_valid_mapping(self) -> None:
        payload = {
            "cases": [
                _valid_hijack_case("c1", hijack_insert="PWNED_BY_HIJACK"),
                _valid_hijack_case("c2", hijack_insert="DATA_EXFIL_OK"),
            ],
        }
        validated = validate_cases_payload(payload, expected_count=2)
        assert len(validated) == 2
        assert validated[0]["hijack_insert"] == "PWNED_BY_HIJACK"

    def test_validate_cases_payload_rejects_too_few_cases(self) -> None:
        payload = {"cases": [_valid_hijack_case("c1")]}
        with pytest.raises(AutomaticCaseGenerationError, match="Expected at least"):
            validate_cases_payload(payload, expected_count=2)

    def test_validate_cases_payload_deduplicates_hijack_insert(self) -> None:
        payload = {
            "cases": [
                _valid_hijack_case("c1", hijack_insert="PWNED_BY_HIJACK"),
                _valid_hijack_case("c2", hijack_insert="PWNED_BY_HIJACK"),
                _valid_hijack_case(
                    "c3",
                    query="Another question?",
                    hijack_insert="DATA_EXFIL_OK",
                ),
            ],
        }
        validated = validate_cases_payload(payload, expected_count=2)
        assert len(validated) == 2

    def test_validate_cases_payload_rejects_empty_query(self) -> None:
        payload = {"cases": [{**_valid_hijack_case("c1"), "query": ""}]}
        with pytest.raises(AutomaticCaseGenerationError, match="query"):
            validate_cases_payload(payload, expected_count=1)

    def test_validate_cases_payload_rejects_invalid_hijack_insert(self) -> None:
        payload = {
            "cases": [
                {
                    **_valid_hijack_case("c1"),
                    "hijack_insert": "bad-marker",
                },
            ],
        }
        with pytest.raises(AutomaticCaseGenerationError, match="hijack_insert"):
            validate_cases_payload(payload, expected_count=1)
