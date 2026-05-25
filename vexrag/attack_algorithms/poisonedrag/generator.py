import json
from collections.abc import Iterable
from time import perf_counter

from vexrag.attack_algorithms.poison_base.correct_answer import resolve_correct_answer
from vexrag.attack_algorithms.poisonedrag.prompts import (
    PROMPT_VERSION,
    build_poison_candidates_prompt,
)
from vexrag.attack_algorithms.poisonedrag.schema import (
    PoisonedRAGMeta,
    PoisonedRAGRequest,
    PoisonedRAGResult,
)
from vexrag.attack_algorithms.poisonedrag.validation import (
    PoisonedRAGValidationError,
    normalize_adv_texts,
    validate_poison_payload,
)
from vexrag.core.attack_configurator import CorrectAnswerProvider
from vexrag.core.llm import JsonCompletionClient, validate_correct_answer_payload


class PoisonedRAGGenerator:
    def __init__(
        self,
        llm_client: JsonCompletionClient,
        correct_answer_provider: CorrectAnswerProvider | None = None,
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
        return resolve_correct_answer(
            query=request.query,
            provided_answer=request.correct_answer,
            target_style=request.target_style,
            seed=request.seed,
            correct_answer_provider=self.correct_answer_provider,
            llm_client=self.llm_client,
            validate_payload=validate_correct_answer_payload,
            warnings=warnings,
        )

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
            poison_candidates_prompt
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
        if request.target_incorrect_answer:
            target_claim = request.target_incorrect_answer.strip()
            if target_claim:
                if request.poisoning_style == "aggressive":
                    adv_texts = [
                        _ensure_target_claim(text, target_claim) for text in adv_texts
                    ]
                elif request.poisoning_style == "soft":
                    adv_texts = [
                        _ensure_soft_claim_hint(text, target_claim)
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


def _ensure_target_claim(text: str, target_claim: str) -> str:
    normalized_text = text.casefold()
    normalized_claim = target_claim.casefold()
    if normalized_claim in normalized_text:
        return text
    separator = "" if text.endswith((" ", "\n", "\t")) else " "
    return f"{text}{separator}{target_claim}".strip()


def _ensure_soft_claim_hint(text: str, target_claim: str) -> str:
    normalized_text = text.casefold()
    if target_claim.casefold() in normalized_text:
        return text
    hint = _build_soft_claim_hint(target_claim)
    if hint.casefold() in normalized_text:
        return text
    separator = "" if text.endswith((" ", "\n", "\t")) else " "
    return f"{text}{separator}{hint}".strip()


def _build_soft_claim_hint(target_claim: str) -> str:
    keywords = [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in target_claim.split()
        if len(token.strip(".,:;!?()[]{}\"'")) >= 5
    ]
    unique_keywords = list(dict.fromkeys(keywords))[:6]
    if not unique_keywords:
        return "Some sources frame the answer in a more controversial way."
    joined = ", ".join(unique_keywords)
    return f"Some sources frame this with cues like: {joined}."
