import pytest

from vexrag.core.evaluation import JudgeAnswerLabel, parse_judge_llm_response
from vexrag.core.evaluation.errors import JudgeResponseValidationError


def test_parse_judge_llm_response_accepts_label_and_reason() -> None:
    details = parse_judge_llm_response(
        {
            "reason": "Matches poison.",
            "judge_answer_label": "attack",
        }
    )
    assert details.label == JudgeAnswerLabel.ATTACK
    assert details.reason == "Matches poison."


def test_parse_judge_llm_response_strips_reason() -> None:
    details = parse_judge_llm_response(
        {
            "reason": "  ok  ",
            "judge_answer_label": "clean",
        }
    )
    assert details.reason == "ok"


def test_parse_judge_llm_response_rejects_missing_reason() -> None:
    with pytest.raises(JudgeResponseValidationError, match="reason"):
        parse_judge_llm_response(
            {
                "judge_answer_label": "clean",
            }
        )


def test_parse_judge_llm_response_rejects_invalid_label() -> None:
    with pytest.raises(JudgeResponseValidationError, match="judge_answer_label"):
        parse_judge_llm_response(
            {
                "reason": "ok",
                "judge_answer_label": "maybe",
            }
        )
