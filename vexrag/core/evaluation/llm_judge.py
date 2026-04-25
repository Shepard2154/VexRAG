import json
from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal, TypedDict

from vexrag.core.evaluation.protocols import (
    EvaluationInput,
    EvaluationResult,
    JudgeLLMProtocol,
    JudgePromptBuilderProtocol,
)

JudgeAnswerLabel = Literal["clean", "attack", "unrelated", "inconclusive"]
JUDGE_ANSWER_LABEL_VALUES = frozenset({"clean", "attack", "unrelated", "inconclusive"})


class JudgeResponse(TypedDict):
    attack_successful: bool
    confidence: float
    reason: str
    judge_answer_label: JudgeAnswerLabel


class JudgeResponseValidationError(ValueError):
    """Raised when an LLM judge response does not match the expected schema."""


class LLMJudgeEvaluator:
    """Evaluates attack success with an LLM judge returning structured JSON."""

    strategy = "llm_judge"

    def __init__(
        self,
        judge_client: JudgeLLMProtocol,
        prompt_builder: JudgePromptBuilderProtocol,
    ) -> None:
        self.judge_client = judge_client
        self.prompt_builder = prompt_builder

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = self.prompt_builder.build_prompt(evaluation_input)
        try:
            raw_response = self.judge_client.complete_json(prompt)
        except Exception as exc:
            return self._failed_result(f"LLM judge request failed: {exc}")

        try:
            verdict = validate_judge_response(raw_response)
        except JudgeResponseValidationError as exc:
            return self._failed_result(
                f"LLM judge response validation failed: {exc}",
                raw_response=raw_response,
            )

        return EvaluationResult(
            attack_successful=verdict["attack_successful"],
            strategy=self.strategy,
            scores={"confidence": verdict["confidence"]},
            reason=verdict["reason"],
            raw_response=raw_response,
            warnings=(),
        )

    def _failed_result(
        self,
        warning: str,
        *,
        raw_response: str | Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=False,
            strategy=self.strategy,
            reason=warning,
            raw_response=raw_response,
            warnings=(warning,),
        )


def validate_judge_response(
    response: str | Mapping[str, Any],
) -> JudgeResponse:
    """Validate and normalize the generic LLM judge JSON response."""
    data = _coerce_response_to_mapping(response)

    attack_successful = data.get("attack_successful")
    if not isinstance(attack_successful, bool):
        raise JudgeResponseValidationError(
            "Field 'attack_successful' must be a boolean."
        )

    confidence = data.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise JudgeResponseValidationError(
            "Field 'confidence' must be a finite number between 0 and 1."
        )
    confidence = float(confidence)

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeResponseValidationError("Field 'reason' must be a non-empty string.")

    judge_answer_label = data.get("judge_answer_label")
    if judge_answer_label not in JUDGE_ANSWER_LABEL_VALUES:
        allowed = "', '".join(sorted(JUDGE_ANSWER_LABEL_VALUES))
        raise JudgeResponseValidationError(
            f"Field 'judge_answer_label' must be one of '{allowed}'."
        )

    return {
        "attack_successful": attack_successful,
        "confidence": confidence,
        "reason": reason.strip(),
        "judge_answer_label": judge_answer_label,
    }


def _coerce_response_to_mapping(response: object) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise JudgeResponseValidationError(
                "LLM response is not valid JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise JudgeResponseValidationError("LLM response must be a JSON object.")
        return parsed
    raise JudgeResponseValidationError("Unsupported LLM response payload type.")
