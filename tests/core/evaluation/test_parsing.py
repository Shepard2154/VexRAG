import pytest

from vexrag.core.evaluation import JudgeAnswerLabel, parse_judge_llm_response
from vexrag.core.evaluation.errors import JudgeResponseValidationError
from vexrag.core.llm.json_validation import (
    LLMPayloadValidationError,
    coerce_payload_to_dict,
    validate_correct_answer_payload,
)


class TestJudgeResponseParsing:
    def test_parse_judge_llm_response_accepts_label_and_reason(self) -> None:
        details = parse_judge_llm_response(
            {
                "reason": "Matches poison.",
                "judge_answer_label": "attack",
            }
        )
        assert details.label == JudgeAnswerLabel.ATTACK
        assert details.reason == "Matches poison."

    def test_parse_judge_llm_response_strips_reason(self) -> None:
        details = parse_judge_llm_response(
            {
                "reason": "  ok  ",
                "judge_answer_label": "clean",
            }
        )
        assert details.reason == "ok"

    def test_parse_judge_llm_response_rejects_missing_reason(self) -> None:
        with pytest.raises(JudgeResponseValidationError, match="reason"):
            parse_judge_llm_response(
                {
                    "judge_answer_label": "clean",
                }
            )

    def test_parse_judge_llm_response_rejects_invalid_label(self) -> None:
        with pytest.raises(JudgeResponseValidationError, match="judge_answer_label"):
            parse_judge_llm_response(
                {
                    "reason": "ok",
                    "judge_answer_label": "maybe",
                }
            )


class TestJsonPayloadValidation:
    def test_coerce_payload_accepts_dict(self) -> None:
        assert coerce_payload_to_dict({"a": 1}) == {"a": 1}

    def test_coerce_payload_parses_json_string(self) -> None:
        assert coerce_payload_to_dict('{"x": 2}') == {"x": 2}

    def test_coerce_payload_rejects_non_object_json(self) -> None:
        with pytest.raises(LLMPayloadValidationError, match="JSON object"):
            coerce_payload_to_dict("[1, 2]")

    def test_coerce_payload_rejects_invalid_json(self) -> None:
        with pytest.raises(LLMPayloadValidationError, match="valid JSON"):
            coerce_payload_to_dict("{not json}")

    def test_coerce_payload_rejects_unsupported_type(self) -> None:
        with pytest.raises(LLMPayloadValidationError, match="Unsupported"):
            coerce_payload_to_dict(42)

    def test_validate_correct_answer_accepts_primary_key(self) -> None:
        assert validate_correct_answer_payload({"correct_answer": "  yes  "}) == "yes"

    def test_validate_correct_answer_accepts_camel_case_alias(self) -> None:
        assert validate_correct_answer_payload({"correctAnswer": "ok"}) == "ok"

    def test_validate_correct_answer_rejects_empty(self) -> None:
        with pytest.raises(LLMPayloadValidationError, match="correct_answer"):
            validate_correct_answer_payload({"correct_answer": "   "})
