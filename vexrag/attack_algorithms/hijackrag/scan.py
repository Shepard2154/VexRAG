import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.attack_algorithms.hijackrag.generator import HijackRAGGenerator
from vexrag.attack_algorithms.hijackrag.report import (
    HijackRAGCaseResult,
    HijackRAGScanReport,
)
from vexrag.attack_algorithms.hijackrag.schema import HijackRAGRequest, HijackRAGResult
from vexrag.core.evaluation import EvaluationInput, EvaluationStrategyProtocol
from vexrag.core.retrieval import CorpusPoisoningAdapterProtocol
from vexrag.core.scan import ScanVerdict
from vexrag.core.target import (
    TargetSystemAdapterProtocol,
    TargetSystemQuery,
    TargetSystemResponse,
)

LOGGER = logging.getLogger("vexrag.scan.hijackrag")


@dataclass(frozen=True, slots=True)
class HijackRAGScanConfig:
    repetitions: int = 1
    attack_success_rate_threshold: float = 0.0
    override_contexts: bool = False
    cleanup: bool = False

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0 <= self.attack_success_rate_threshold <= 1:
            raise ValueError("attack_success_rate_threshold must be between 0 and 1")


class HijackRAGScanRunner:
    def __init__(
        self,
        generator: HijackRAGGenerator,
        target_system: TargetSystemAdapterProtocol,
        evaluation_strategy: EvaluationStrategyProtocol,
        corpus_poisoner: CorpusPoisoningAdapterProtocol | None = None,
    ) -> None:
        self.generator = generator
        self.target_system = target_system
        self.evaluation_strategy = evaluation_strategy
        self.corpus_poisoner = corpus_poisoner

    def run(
        self,
        requests: Sequence[HijackRAGRequest],
        config: HijackRAGScanConfig | None = None,
    ) -> HijackRAGScanReport:
        scan_config = config or HijackRAGScanConfig()
        if not requests:
            raise ValueError("at least one HijackRAG case is required")

        cases: list[HijackRAGCaseResult] = []
        generated_results: list[HijackRAGResult] = []
        for case_index, request in enumerate(requests, start=1):
            case_label = request.case_id or f"#{case_index}"
            LOGGER.info("Building HijackRAG adversarial texts for case %s", case_label)
            generated = self.generator.generate_one(request)
            LOGGER.info(
                "Built %d adversarial text(s) for case %s",
                len(generated.adv_texts),
                case_label,
            )
            generated_results.append(generated)
            cases.extend(self._run_cases(generated, request, case_index, scan_config))

        return self._build_report(tuple(generated_results), tuple(cases), scan_config)

    def _run_cases(
        self,
        generated: HijackRAGResult,
        request: HijackRAGRequest,
        case_index: int,
        config: HijackRAGScanConfig,
    ) -> tuple[HijackRAGCaseResult, ...]:
        results: list[HijackRAGCaseResult] = []
        for run_index in range(1, config.repetitions + 1):
            LOGGER.info(
                "Querying target for case %s, run %d, hijack texts %d",
                request.case_id or f"#{case_index}",
                run_index,
                len(generated.adv_texts),
            )
            results.append(
                self._run_case(
                    generated=generated,
                    request=request,
                    adversarial_texts=tuple(generated.adv_texts),
                    case_index=case_index,
                    run_index=run_index,
                    override_contexts=config.override_contexts,
                    cleanup=config.cleanup,
                )
            )
        return tuple(results)

    def _run_case(
        self,
        *,
        generated: HijackRAGResult,
        request: HijackRAGRequest,
        adversarial_texts: tuple[str, ...],
        case_index: int,
        run_index: int,
        override_contexts: bool,
        cleanup: bool,
    ) -> HijackRAGCaseResult:
        metadata = self._case_metadata(
            request=request,
            generated=generated,
            case_index=case_index,
            run_index=run_index,
            adversarial_text_count=len(adversarial_texts),
        )
        poisoned_document_ids = self._poison_retrieval_corpus(
            adversarial_texts=adversarial_texts,
            metadata=metadata,
        )
        if poisoned_document_ids:
            metadata["poisoned_document_ids"] = poisoned_document_ids
        try:
            system_response = self._query_target_system(
                generated=generated,
                adversarial_texts=adversarial_texts,
                metadata=metadata,
                override_contexts=override_contexts,
            )
            evaluation_input = self._build_evaluation_input(
                generated=generated,
                adversarial_texts=adversarial_texts,
                system_response=system_response,
                metadata=metadata,
                override_contexts=override_contexts,
            )
            evaluation = self.evaluation_strategy.evaluate(evaluation_input)
        finally:
            self._cleanup_poisoned_documents(
                poisoned_document_ids=poisoned_document_ids,
                cleanup=cleanup,
            )
        return HijackRAGCaseResult(
            query=generated.query,
            adversarial_texts=adversarial_texts,
            expected_incorrect_answer=generated.incorrect_answer,
            system_response=system_response,
            evaluation=evaluation,
            case_id=request.case_id,
            run_index=run_index,
        )

    def _query_target_system(
        self,
        *,
        generated: HijackRAGResult,
        adversarial_texts: tuple[str, ...],
        metadata: dict[str, Any],
        override_contexts: bool,
    ) -> TargetSystemResponse:
        contexts = adversarial_texts if override_contexts else ()
        return self.target_system.answer(
            TargetSystemQuery(
                query=generated.query,
                contexts=contexts,
                metadata=metadata,
            )
        )

    def _build_evaluation_input(
        self,
        *,
        generated: HijackRAGResult,
        adversarial_texts: tuple[str, ...],
        system_response: TargetSystemResponse,
        metadata: dict[str, Any],
        override_contexts: bool,
    ) -> EvaluationInput:
        return EvaluationInput(
            query=generated.query,
            actual_answer=system_response.answer,
            expected_clean_answer=generated.correct_answer,
            expected_attack_answer=generated.incorrect_answer,
            retrieved_contexts=system_response.contexts,
            context_override=adversarial_texts if override_contexts else None,
            metadata=metadata,
        )

    def _build_report(
        self,
        generated_results: tuple[HijackRAGResult, ...],
        cases: tuple[HijackRAGCaseResult, ...],
        config: HijackRAGScanConfig,
    ) -> HijackRAGScanReport:
        successful_cases = sum(case.successful for case in cases)
        success_rate = successful_cases / len(cases) if cases else 0.0
        vulnerable = successful_cases > 0 and (
            success_rate >= config.attack_success_rate_threshold
        )

        warnings = tuple(
            warning
            for generated in generated_results
            for warning in generated.meta.warnings
        )
        if not cases:
            warnings = (*warnings, "No adversarial texts were generated for scanning.")

        return HijackRAGScanReport(
            verdict=(
                ScanVerdict.VULNERABLE if vulnerable else ScanVerdict.NOT_VULNERABLE
            ),
            cases=tuple(cases),
            generated_adversarial_texts=tuple(
                adversarial_text
                for generated in generated_results
                for adversarial_text in generated.adv_texts
            ),
            warnings=warnings,
        )

    @staticmethod
    def _case_metadata(
        request: HijackRAGRequest,
        generated: HijackRAGResult,
        case_index: int,
        run_index: int,
        adversarial_text_count: int,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "attack": "hijackrag",
            "case_index": case_index,
            "run_index": run_index,
            "adversarial_text_count": adversarial_text_count,
            "query": generated.query,
            "expected_clean_answer": generated.correct_answer,
            "expected_attack_answer": generated.incorrect_answer,
            "hijack_segment_ids": list(generated.meta.segment_ids),
        }
        if request.case_id:
            metadata["case_id"] = request.case_id
        return metadata

    def _poison_retrieval_corpus(
        self,
        *,
        adversarial_texts: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        if self.corpus_poisoner is None or not adversarial_texts:
            return ()
        LOGGER.info(
            "Writing %d adversarial text(s) into retrieval corpus",
            len(adversarial_texts),
        )
        return self.corpus_poisoner.add_texts(adversarial_texts, metadata)

    def _cleanup_poisoned_documents(
        self,
        *,
        poisoned_document_ids: tuple[str, ...],
        cleanup: bool,
    ) -> None:
        if not cleanup or not poisoned_document_ids or self.corpus_poisoner is None:
            return
        LOGGER.info("Cleaning up hijack texts from retrieval corpus")
        self.corpus_poisoner.delete_texts(poisoned_document_ids)
