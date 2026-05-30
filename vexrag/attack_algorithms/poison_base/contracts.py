from collections.abc import Iterable
from typing import Protocol, TypeVar

RequestT = TypeVar("RequestT", bound="CorpusPoisonRequest")


class CorpusPoisonRequest(Protocol):
    case_id: str | None


class CorpusPoisonGenerationMeta(Protocol):
    warnings: list[str]


class CorpusPoisonGenerationResult(Protocol):
    query: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: list[str]
    meta: CorpusPoisonGenerationMeta


class CorpusPoisonGenerator(Protocol):
    def generate_one(
        self, request: CorpusPoisonRequest
    ) -> CorpusPoisonGenerationResult: ...

    def generate_many(
        self, requests: Iterable[CorpusPoisonRequest]
    ) -> list[CorpusPoisonGenerationResult]: ...
