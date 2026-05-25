from collections.abc import Callable
from typing import Any

from vexrag.core.attack_configurator import CorrectAnswerProvider, TargetStyle
from vexrag.core.llm import JsonCompletionClient, build_correct_answer_prompt


def resolve_correct_answer(
    *,
    query: str,
    provided_answer: str | None,
    target_style: TargetStyle,
    seed: int | None,
    correct_answer_provider: CorrectAnswerProvider | None,
    llm_client: JsonCompletionClient,
    validate_payload: Callable[[Any], str],
    warnings: list[str],
) -> tuple[str, str]:
    """Resolve correct_answer and its source using one of: provided, target-system, or LLM fallback."""
    if provided_answer and provided_answer.strip():
        return provided_answer.strip(), "provided"

    if correct_answer_provider is not None:
        candidate = correct_answer_provider.get_correct_answer(
            query,
            target_style=target_style,
            seed=seed,
        )
        text = candidate.strip()
        if text:
            return text, "target_system"
        warnings.append(
            "Target system returned empty correct answer; using LLM fallback."
        )

    payload = llm_client.complete_json(
        build_correct_answer_prompt(
            query=query,
            target_style=target_style,
        ),
    )
    return validate_payload(payload), "llm_generated"
