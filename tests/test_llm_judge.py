import pytest

from vexrag.core.evaluation.llm_judge import (
    JudgeResponseValidationError,
    validate_judge_response,
)


def test_validate_judge_response_accepts_strict_boolean() -> None:
    verdict = validate_judge_response(
        {
            "attack_successful": True,
            "reason": "Matches poison.",
            "judge_answer_label": "attack",
        }
    )
    assert verdict["attack_successful"] is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("FALSE", False),
        ("yes", True),
        ("no", False),
        ("1", True),
        ("0", False),
        (1, True),
        (0, False),
    ],
)
def test_validate_judge_response_coerces_attack_successful(
    raw: object, expected: bool
) -> None:
    verdict = validate_judge_response(
        {
            "attack_successful": raw,
            "reason": "ok",
            "judge_answer_label": "clean",
        }
    )
    assert verdict["attack_successful"] is expected


def test_validate_judge_response_rejects_invalid_attack_successful() -> None:
    with pytest.raises(JudgeResponseValidationError, match="attack_successful"):
        validate_judge_response(
            {
                "attack_successful": "maybe",
                "reason": "ok",
                "judge_answer_label": "clean",
            }
        )
