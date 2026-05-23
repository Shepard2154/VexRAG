from vexrag.core.evaluation.contracts import JudgePromptBuilder
from vexrag.core.evaluation.errors import EvaluatorError
from vexrag.core.evaluation.parsing import parse_judge_llm_response
from vexrag.core.evaluation.types import (
    EvaluationInput,
    EvaluationResult,
    EvaluationStrategy,
    attack_successful_from_judge_label,
    make_incomplete_verdict,
)
from vexrag.core.llm.contracts import JsonCompletionClient


class LLMJudgeEvaluator:
    """Evaluates attack success with an LLM judge returning structured JSON."""

    strategy = EvaluationStrategy.LLM_JUDGE

    def __init__(
        self,
        judge_client: JsonCompletionClient,
        prompt_builder: JudgePromptBuilder,
    ) -> None:
        self.judge_client = judge_client
        self.prompt_builder = prompt_builder

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = self.prompt_builder.build_prompt(evaluation_input)
        try:
            raw_response = self.judge_client.complete_json(prompt)
        except EvaluatorError as err:
            return make_incomplete_verdict(self.strategy, str(err))
        try:
            return self._evaluate_from_raw(raw_response)
        except EvaluatorError as err:
            return make_incomplete_verdict(
                self.strategy,
                str(err),
                raw=raw_response,
            )

    def _evaluate_from_raw(self, raw_response: object) -> EvaluationResult:
        judge = parse_judge_llm_response(raw_response)
        attack_successful = attack_successful_from_judge_label(judge.label)
        return EvaluationResult(
            attack_successful=attack_successful,
            completed=True,
            strategy=self.strategy,
            reason=judge.reason,
            judge=judge,
            raw=raw_response,
            warnings=(),
        )
