from collections.abc import Mapping
from typing import Any

from vexrag.core.evaluation.errors import JudgeResponseValidationError
from vexrag.core.evaluation.types import JudgeAnswerLabel, JudgeDetails
from vexrag.core.llm.json_validation import (
    LLMPayloadValidationError,
    coerce_payload_to_dict,
)


def parse_judge_llm_response(payload: str | Mapping[str, Any]) -> JudgeDetails:
    try:
        data = coerce_payload_to_dict(payload)
    except LLMPayloadValidationError as exc:
        raise JudgeResponseValidationError(str(exc)) from exc

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeResponseValidationError("Field 'reason' must be a non-empty string.")

    judge_answer_label = data.get("judge_answer_label")
    try:
        label = JudgeAnswerLabel(judge_answer_label)
    except ValueError:
        allowed = "', '".join(sorted(m.value for m in JudgeAnswerLabel))
        raise JudgeResponseValidationError(
            f"Field 'judge_answer_label' must be one of '{allowed}'."
        ) from None

    return JudgeDetails(label=label, reason=reason.strip())
