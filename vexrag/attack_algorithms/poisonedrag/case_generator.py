from vexrag.attack_algorithms.poisonedrag.prompts.automatic_cases import (
    PROMPT_VERSION,
    build_automatic_cases_prompt,
)
from vexrag.attack_algorithms.poisonedrag.scan_profile import POISONEDRAG_SCAN_PROFILE
from vexrag.attack_algorithms.poisonedrag.schema import PoisonedRAGRequest
from vexrag.attack_algorithms.poisonedrag.validation import (
    AutomaticCaseGenerationError,
    validate_cases_payload,
)
from vexrag.core.attack_configurator import TargetStyle
from vexrag.core.llm import JsonCompletionClient


class AutomaticPoisonedRAGCaseGenerator:
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
        target_style: TargetStyle = "short_fact",
        adv_per_query: int = POISONEDRAG_SCAN_PROFILE.default_adv_per_query,
        seed: int | None = None,
    ) -> tuple[PoisonedRAGRequest, ...]:
        del seed
        if count < 1:
            raise AutomaticCaseGenerationError("count must be at least 1")
        if adv_per_query < 1:
            raise AutomaticCaseGenerationError("adv_per_query must be at least 1")

        prompt = build_automatic_cases_prompt(
            count=count,
            topic=topic,
            target_style=target_style,
        )
        payload = self.llm_client.complete_json(prompt)
        cases = validate_cases_payload(payload, expected_count=count)
        return tuple(
            PoisonedRAGRequest(
                query=case["query"],
                correct_answer=case["correct_answer"],
                target_incorrect_answer=case["target_incorrect_answer"],
                case_id=case["id"],
                target_style=target_style,
                adv_per_query=adv_per_query,
            )
            for case in cases
        )
