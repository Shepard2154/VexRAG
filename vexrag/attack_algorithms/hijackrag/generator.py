import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter

from vexrag.attack_algorithms.hijackrag.schema import (
    HijackRAGMeta,
    HijackRAGRequest,
    HijackRAGResult,
)
from vexrag.attack_algorithms.hijackrag.segments import (
    HijackSegmentRecord,
    apply_hijack_insert,
    load_hijack_segments,
)
from vexrag.attack_algorithms.poison_base.correct_answer import resolve_correct_answer
from vexrag.core.attack_configurator import CorrectAnswerProvider
from vexrag.core.llm import JsonCompletionClient, validate_correct_answer_payload

PROMPT_VERSION = "hijackrag-segments-v1"


class HijackRAGGenerator:
    """Builds corpus poison texts from HijackRAG segment templates."""

    def __init__(
        self,
        segments: Sequence[HijackSegmentRecord],
        llm_client: JsonCompletionClient,
        correct_answer_provider: CorrectAnswerProvider | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._segments = tuple(segments)
        self._by_id: dict[str, HijackSegmentRecord] = {
            record.segment_id: record for record in self._segments
        }
        self.llm_client = llm_client
        self.correct_answer_provider = correct_answer_provider
        self.prompt_version = prompt_version

    @classmethod
    def from_segments_file(
        cls,
        path: Path,
        llm_client: JsonCompletionClient,
        correct_answer_provider: CorrectAnswerProvider | None = None,
    ) -> "HijackRAGGenerator":
        return cls(
            load_hijack_segments(path),
            llm_client,
            correct_answer_provider=correct_answer_provider,
        )

    def generate_one(self, request: HijackRAGRequest) -> HijackRAGResult:
        started = perf_counter()
        warnings: list[str] = []

        correct_answer, correct_source = self._resolve_correct_answer(request, warnings)
        insert = request.hijack_insert.strip()
        chosen, used_ids = self._pick_segments(request, warnings)
        adv_texts = self._build_adv_texts(request, insert, chosen, warnings)

        latency_ms = int((perf_counter() - started) * 1000)
        model_id = getattr(self.llm_client, "model_id", "unknown")

        return HijackRAGResult(
            query=request.query,
            correct_answer=correct_answer,
            incorrect_answer=insert,
            adv_texts=adv_texts,
            meta=HijackRAGMeta(
                latency_ms=latency_ms,
                prompt_version=self.prompt_version,
                model_id=model_id,
                correct_answer_source=correct_source,
                segment_ids=used_ids,
                warnings=warnings,
            ),
        )

    def generate_many(
        self,
        requests: Iterable[HijackRAGRequest],
    ) -> list[HijackRAGResult]:
        return [self.generate_one(request) for request in requests]

    def _resolve_correct_answer(
        self,
        request: HijackRAGRequest,
        warnings: list[str],
    ) -> tuple[str, str]:
        return resolve_correct_answer(
            query=request.query,
            provided_answer=request.correct_answer,
            target_style=request.correct_answer_style,
            seed=request.seed,
            correct_answer_provider=self.correct_answer_provider,
            llm_client=self.llm_client,
            validate_payload=validate_correct_answer_payload,
            warnings=warnings,
        )

    @staticmethod
    def _clean_unique_segment_ids(raw_ids: Iterable[str]) -> list[str]:
        cleaned = (sid for sid in (str(raw_id).strip() for raw_id in raw_ids) if sid)
        return list(dict.fromkeys(cleaned))

    def _segments_from_explicit_ids(
        self,
        segment_ids: Iterable[str],
        limit: int,
        warnings: list[str],
    ) -> list[HijackSegmentRecord]:
        chosen: list[HijackSegmentRecord] = []
        for sid in self._clean_unique_segment_ids(segment_ids):
            rec = self._by_id.get(sid)
            if rec is None:
                warnings.append(f"Unknown hijack segment id ignored: {sid}")
                continue
            chosen.append(rec)
            if len(chosen) >= limit:
                break
        return chosen

    def _extend_segments_with_random_fill(
        self,
        chosen: list[HijackSegmentRecord],
        target_count: int,
        rng: random.Random,
        warnings: list[str],
    ) -> None:
        seen = {r.segment_id for r in chosen}
        pool = [r for r in self._segments if r.segment_id not in seen]
        rng.shuffle(pool)
        pool_index = 0
        while len(chosen) < target_count:
            if not pool:
                pool = list(self._segments)
                rng.shuffle(pool)
                pool_index = 0
                if len(self._segments) < target_count:
                    warnings.append(
                        "Fewer unique HijackRAG segments than adv_per_query; reusing templates."
                    )
            chosen.append(pool[pool_index % len(pool)])
            pool_index += 1

    def _pick_segments(
        self,
        request: HijackRAGRequest,
        warnings: list[str],
    ) -> tuple[list[HijackSegmentRecord], tuple[str, ...]]:
        n = max(1, request.adv_per_query)
        chosen = self._segments_from_explicit_ids(request.segment_ids, n, warnings)
        if len(chosen) >= n:
            return chosen, tuple(r.segment_id for r in chosen)

        rng = (
            random.Random(request.seed) if request.seed is not None else random.Random()
        )
        self._extend_segments_with_random_fill(chosen, n, rng, warnings)
        return chosen, tuple(r.segment_id for r in chosen)

    def _build_adv_texts(
        self,
        request: HijackRAGRequest,
        hijack_insert: str,
        segments: Sequence[HijackSegmentRecord],
        warnings: list[str],
    ) -> list[str]:
        texts: list[str] = []
        query_prefix = request.query.strip()
        for rec in segments:
            body = apply_hijack_insert(rec.template, hijack_insert)
            if query_prefix and not body.startswith(query_prefix):
                text = f"{query_prefix}. {body}"
            else:
                text = body
            texts.append(text)
        if len(texts) < request.adv_per_query:
            warnings.append(
                "Generated fewer adversarial texts than requested after segment selection."
            )
        return texts
