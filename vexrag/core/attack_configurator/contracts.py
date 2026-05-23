from typing import Protocol

from vexrag.core.attack_configurator.types import TargetStyle


class PoisonedResult(Protocol):
    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]


class CorrectAnswerProvider(Protocol):
    def get_correct_answer(
        self,
        query: str,
        *,
        target_style: TargetStyle,
        seed: int | None = None,
    ) -> str: ...
