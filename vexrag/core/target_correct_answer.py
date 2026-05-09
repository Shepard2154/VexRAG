from vexrag.core.target import HTTPTargetSystemAdapter, TargetSystemQuery


class TargetCorrectAnswerProvider:
    """Fetches correct answers from the configured HTTP target RAG."""
    __slots__ = ("target_system", "attack")

    def __init__(self, target_system: HTTPTargetSystemAdapter, attack: str) -> None:
        self.target_system = target_system
        self.attack = attack

    def get_correct_answer(
        self,
        query: str,
        *,
        target_style: str,
        seed: int | None = None,
    ) -> str:
        response = self.target_system.answer(
            TargetSystemQuery(
                query=query,
                metadata={
                    "attack": self.attack,
                    "purpose": "correct_answer",
                    "target_style": target_style,
                    "seed": seed,
                },
            )
        )
        return response.answer
