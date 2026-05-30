import pytest

from vexrag.attack_algorithms.poison_base.case_validation import (
    normalize_case_id,
    required_text,
)


class _CaseError(ValueError):
    pass


class TestCaseValidation:
    def test_required_text_strips_and_returns(self) -> None:
        assert required_text("  hello  ", field="q", error_type=_CaseError) == "hello"

    def test_required_text_rejects_non_string(self) -> None:
        with pytest.raises(_CaseError, match="q"):
            required_text(1, field="q", error_type=_CaseError)

    def test_required_text_rejects_blank(self) -> None:
        with pytest.raises(_CaseError, match="q"):
            required_text("   ", field="q", error_type=_CaseError)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("My Case!", "my_case"),
            ("---", "generated_case"),
            ("A" * 80, "a" * 64),
        ],
    )
    def test_normalize_case_id(self, raw: str, expected: str) -> None:
        assert normalize_case_id(raw) == expected
