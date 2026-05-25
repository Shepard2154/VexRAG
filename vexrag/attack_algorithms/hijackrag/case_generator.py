from vexrag.attack_algorithms.hijackrag.prompts.automatic_cases import (
    PROMPT_VERSION,
    build_automatic_cases_prompt,
)
from vexrag.attack_algorithms.hijackrag.schema import HijackRAGRequest
from vexrag.attack_algorithms.hijackrag.validation import (
    AutomaticCaseGenerationError,
    validate_cases_payload,
)
from vexrag.core.attack_configurator import TargetStyle
from vexrag.core.llm import JsonCompletionClient


class AutomaticHijackRAGCaseGenerator:
    def __init__(
        self,
        llm_client: JsonCompletionClient,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_version = prompt_version

    def generate_cases(
        self,
        *,
        count: int,
        topic: str | None = None,
        correct_answer_style: TargetStyle = "short_fact",
        adv_per_query: int = 1,
        seed: int | None = None,
    ) -> tuple[HijackRAGRequest, ...]:
        if count < 1:
            raise AutomaticCaseGenerationError("count must be at least 1")
        if adv_per_query < 1:
            raise AutomaticCaseGenerationError("adv_per_query must be at least 1")

        prompt = build_automatic_cases_prompt(
            count=count,
            topic=topic,
            correct_answer_style=correct_answer_style,
        )
        payload = self.llm_client.complete_json(prompt)
        cases = validate_cases_payload(payload, expected_count=count)
        requests: list[HijackRAGRequest] = []
        for offset, case in enumerate(cases):
            case_seed = None if seed is None else int(seed) + offset
            requests.append(
                HijackRAGRequest(
                    query=case["query"],
                    hijack_insert=case["hijack_insert"],
                    correct_answer=case["correct_answer"],
                    case_id=case["id"],
                    adv_per_query=adv_per_query,
                    segment_ids=(),
                    correct_answer_style=correct_answer_style,
                    seed=case_seed,
                )
            )
        return tuple(requests)
