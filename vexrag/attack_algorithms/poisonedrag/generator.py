import json
from collections.abc import Iterable
from time import perf_counter
from typing import Any, Protocol

from vexrag.attack_algorithms.poisonedrag.prompts import (
    PROMPT_VERSION,
    build_correct_answer_prompt,
    build_poison_candidates_prompt,
)
from vexrag.attack_algorithms.poisonedrag.schema import (
    PoisonedRAGMeta,
    PoisonedRAGRequest,
    PoisonedRAGResult,
    TargetStyle,
)
from vexrag.attack_algorithms.poisonedrag.validators import (
    PoisonedRAGValidationError,
    normalize_adv_texts,
    validate_correct_answer_payload,
    validate_poison_payload,
)


class LLMClientProtocol(Protocol):
    model_id: str

    def complete_json(
        self,
        prompt: str,
        *,
        schema_name: str | None = None,
        seed: int | None = None,
    ) -> str | dict[str, Any]: ...


class CorrectAnswerProviderProtocol(Protocol):
    def get_correct_answer(
        self,
        query: str,
        *,
        target_style: TargetStyle,
        seed: int | None = None,
    ) -> str: ...


class PoisonedRAGGenerator:
    def __init__(
        self,
        llm_client: LLMClientProtocol,
        correct_answer_provider: CorrectAnswerProviderProtocol | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.llm_client = llm_client
        self.correct_answer_provider = correct_answer_provider
        self.prompt_version = prompt_version

    def generate_one(self, request: PoisonedRAGRequest) -> PoisonedRAGResult:
        started = perf_counter()
        warnings: list[str] = []

        correct_answer, correct_answer_source = self._resolve_correct_answer(
            request,
            warnings,
        )
        incorrect_answer, adv_texts = self._generate_poison_outputs(
            request,
            correct_answer,
            warnings,
        )

        latency_ms = int((perf_counter() - started) * 1000)
        model_id = getattr(self.llm_client, "model_id", "unknown")

        return PoisonedRAGResult(
            query=request.query,
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
            adv_texts=adv_texts,
            meta=PoisonedRAGMeta(
                latency_ms=latency_ms,
                prompt_version=self.prompt_version,
                model_id=model_id,
                correct_answer_source=correct_answer_source,
                warnings=warnings,
            ),
        )

    def _resolve_correct_answer(
        self,
        request: PoisonedRAGRequest,
        warnings: list[str],
    ) -> tuple[str, str]:
        correct_answer = request.correct_answer
        correct_answer_source = "provided"
        if correct_answer:
            return correct_answer, correct_answer_source

        if self.correct_answer_provider is not None:
            candidate = self.correct_answer_provider.get_correct_answer(
                request.query,
                target_style=request.target_style,
                seed=request.seed,
            )
            correct_answer = candidate.strip()
            if correct_answer:
                return correct_answer, "target_system"
            warnings.append(
                "Target system returned empty correct answer; using LLM fallback."
            )

        correct_answer_prompt = build_correct_answer_prompt(
            query=request.query,
            target_style=request.target_style,
        )
        correct_answer_payload = self.llm_client.complete_json(
            correct_answer_prompt,
            schema_name="poisonedrag.stage1.correct_answer",
            seed=request.seed,
        )
        correct_answer = validate_correct_answer_payload(correct_answer_payload)
        return correct_answer, "llm_generated"

    def _generate_poison_outputs(
        self,
        request: PoisonedRAGRequest,
        correct_answer: str,
        warnings: list[str],
    ) -> tuple[str, list[str]]:
        poison_candidates_prompt = build_poison_candidates_prompt(
            query=request.query,
            correct_answer=correct_answer,
            target_incorrect_answer=request.target_incorrect_answer,
            adv_per_query=request.adv_per_query,
            target_style=request.target_style,
        )
        poison_candidates_payload = self.llm_client.complete_json(
            poison_candidates_prompt,
            schema_name="poisonedrag.stage2.poison_candidates",
            seed=request.seed,
        )
        try:
            incorrect_answer, raw_adv_texts = validate_poison_payload(
                poison_candidates_payload
            )
        except PoisonedRAGValidationError as exc:
            raise PoisonedRAGValidationError(
                f"{exc} Raw response: {_format_raw_payload(poison_candidates_payload)}"
            ) from exc
        if request.target_incorrect_answer:
            target_incorrect = request.target_incorrect_answer.strip()
            if target_incorrect and incorrect_answer != target_incorrect:
                warnings.append(
                    "Model returned a different incorrect answer; forcing target_incorrect_answer."
                )
                incorrect_answer = target_incorrect

        adv_texts = normalize_adv_texts(
            raw_adv_texts,
            limit=request.adv_per_query,
            seed=request.seed,
        )
        query_prefix = request.query.strip()
        if query_prefix:
            adv_texts = [
                f"{query_prefix}. {text}" if not text.startswith(query_prefix) else text
                for text in adv_texts
            ]
        if len(adv_texts) < request.adv_per_query:
            warnings.append(
                "Generated fewer adversarial texts than requested after validation."
            )
        return incorrect_answer, adv_texts

    def generate_many(
        self,
        requests: Iterable[PoisonedRAGRequest],
    ) -> list[PoisonedRAGResult]:
        return [self.generate_one(request) for request in requests]


def _format_raw_payload(payload: object, *, limit: int = 2000) -> str:
    if isinstance(payload, dict):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    else:
        text = str(payload)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}... [truncated {omitted} chars]"
