import pytest

from vexrag.attack_algorithms.poisonedrag.validation import (
    AutomaticCaseGenerationError,
    validate_cases_payload,
)


def _valid_poison_case(case_id: str, query: str = "What is RAG?") -> dict[str, str]:
    return {
        "id": case_id,
        "query": query,
        "correct_answer": "Clean answer.",
        "target_incorrect_answer": "Wrong answer.",
    }


class TestPoisonedRagValidateCasesPayload:
    def test_validate_cases_payload_accepts_valid_mapping(self) -> None:
        payload = {"cases": [_valid_poison_case("c1"), _valid_poison_case("c2")]}
        validated = validate_cases_payload(payload, expected_count=2)
        assert len(validated) == 2
        assert validated[0]["id"] == "c1"
        assert validated[0]["query"] == "What is RAG?"

    def test_validate_cases_payload_rejects_too_few_cases(self) -> None:
        payload = {"cases": [_valid_poison_case("c1")]}
        with pytest.raises(AutomaticCaseGenerationError, match="Expected at least"):
            validate_cases_payload(payload, expected_count=2)

    def test_validate_cases_payload_deduplicates_case_id(self) -> None:
        payload = {
            "cases": [
                _valid_poison_case("dup"),
                {**_valid_poison_case("dup"), "query": "Other query"},
                _valid_poison_case("unique"),
            ],
        }
        validated = validate_cases_payload(payload, expected_count=2)
        assert len(validated) == 2
        assert {case["id"] for case in validated} == {"dup", "unique"}

    def test_validate_cases_payload_rejects_empty_query(self) -> None:
        payload = {"cases": [{**_valid_poison_case("c1"), "query": "   "}]}
        with pytest.raises(AutomaticCaseGenerationError, match="query"):
            validate_cases_payload(payload, expected_count=1)
