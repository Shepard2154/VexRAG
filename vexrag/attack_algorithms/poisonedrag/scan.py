import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vexrag.attack_algorithms.poisonedrag.generator import PoisonedRAGGenerator
from vexrag.attack_algorithms.poisonedrag.report import (
    PoisonedRAGCaseResult,
    PoisonedRAGScanReport,
)
from vexrag.attack_algorithms.poisonedrag.schema import (
    PoisonedRAGRequest,
    PoisonedRAGResult,
)
from vexrag.core.adversarial_probe import probe_with_poisoning_and_evaluation
from vexrag.core.contracts import ScanVerdict
from vexrag.core.evaluation import EvaluationStrategyProtocol
from vexrag.core.retrieval import CorpusPoisoningAdapterProtocol
from vexrag.core.target import TargetSystemAdapterProtocol

LOGGER = logging.getLogger("vexrag.scan.poisonedrag")


@dataclass(frozen=True, slots=True)
class PoisonedRAGScanConfig:
    """Execution settings for a PoisonedRAG target-system scan."""
    repetitions: int = 1
    attack_success_rate_threshold: float = 0.0
    override_contexts: bool = False
    cleanup: bool = False

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0 <= self.attack_success_rate_threshold <= 1:
            raise ValueError("attack_success_rate_threshold must be between 0 and 1")


class PoisonedRAGScanRunner:
    """Orchestrates PoisonedRAG generation and target-system checks."""
    def __init__(
        self,
        generator: PoisonedRAGGenerator,
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
        requests: Sequence[PoisonedRAGRequest],
        config: PoisonedRAGScanConfig | None = None,
    ) -> PoisonedRAGScanReport:
        scan_config = config or PoisonedRAGScanConfig()
        if not requests:
            raise ValueError("at least one PoisonedRAG case is required")

        cases: list[PoisonedRAGCaseResult] = []
        generated_results: list[PoisonedRAGResult] = []
        for case_index, request in enumerate(requests, start=1):
            case_label = request.case_id or f"#{case_index}"
            LOGGER.info("Generating adversarial texts for case %s", case_label)
            generated = self.generator.generate_one(request)
            LOGGER.info(
                "Generated %d adversarial text(s) for case %s",
                len(generated.adv_texts),
                case_label,
            )
            generated_results.append(generated)
            cases.extend(self._run_cases(generated, request, case_index, scan_config))

        return self._build_report(tuple(generated_results), tuple(cases), scan_config)

    def _run_cases(
        self,
        generated: PoisonedRAGResult,
        request: PoisonedRAGRequest,
        case_index: int,
        config: PoisonedRAGScanConfig,
    ) -> tuple[PoisonedRAGCaseResult, ...]:
        cases: list[PoisonedRAGCaseResult] = []
        for run_index in range(1, config.repetitions + 1):
            LOGGER.info(
                "Querying target for case %s, run %d, poisoned texts %d",
                request.case_id or f"#{case_index}",
                run_index,
                len(generated.adv_texts),
            )
            cases.append(
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
        return tuple(cases)

    def _run_case(
        self,
        *,
        generated: PoisonedRAGResult,
        request: PoisonedRAGRequest,
        adversarial_texts: tuple[str, ...],
        case_index: int,
        run_index: int,
        override_contexts: bool,
        cleanup: bool,
    ) -> PoisonedRAGCaseResult:
        metadata = self._case_metadata(
            request=request,
            generated=generated,
            case_index=case_index,
            run_index=run_index,
            adversarial_text_count=len(adversarial_texts),
        )
        system_response, evaluation = probe_with_poisoning_and_evaluation(
            query=generated.query,
            correct_answer=generated.correct_answer,
            incorrect_answer=generated.incorrect_answer,
            adversarial_texts=adversarial_texts,
            corpus_poisoner=self.corpus_poisoner,
            target_system=self.target_system,
            evaluation_strategy=self.evaluation_strategy,
            override_contexts=override_contexts,
            cleanup=cleanup,
            metadata=metadata,
            corpus_cleanup_label="poisoned texts",
        )
        return PoisonedRAGCaseResult(
            query=generated.query,
            adversarial_texts=adversarial_texts,
            expected_incorrect_answer=generated.incorrect_answer,
            system_response=system_response,
            evaluation=evaluation,
            case_id=request.case_id,
            run_index=run_index,
        )

    def _build_report(
        self,
        generated_results: tuple[PoisonedRAGResult, ...],
        cases: tuple[PoisonedRAGCaseResult, ...],
        config: PoisonedRAGScanConfig,
    ) -> PoisonedRAGScanReport:
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

        return PoisonedRAGScanReport(
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
        request: PoisonedRAGRequest,
        generated: PoisonedRAGResult,
        case_index: int,
        run_index: int,
        adversarial_text_count: int,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "attack": "poisonedrag",
            "case_index": case_index,
            "run_index": run_index,
            "adversarial_text_count": adversarial_text_count,
            "query": generated.query,
            "expected_clean_answer": generated.correct_answer,
            "expected_attack_answer": generated.incorrect_answer,
        }
        if request.case_id:
            metadata["case_id"] = request.case_id
        return metadata
