import pytest

from vexrag.core.evaluation.attack_verdict import JudgeAnswerLabel
from vexrag.core.evaluation.errors import JudgeResponseValidationError
from vexrag.core.evaluation.judge_response_parser import parse_judge_llm_response


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
