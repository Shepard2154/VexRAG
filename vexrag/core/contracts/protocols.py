from typing import Any, Protocol

from vexrag.core.contracts.attack_literals import TargetStyle


class PoisonedResult(Protocol):
    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]


class LLMClientProtocol(Protocol):
    model_id: str

    def complete_json(self, prompt: str) -> str | dict[str, Any]: ...


class CorrectAnswerProviderProtocol(Protocol):
    def get_correct_answer(
        self,
        query: str,
        *,
        target_style: TargetStyle,
        seed: int | None = None,
    ) -> str: ...
