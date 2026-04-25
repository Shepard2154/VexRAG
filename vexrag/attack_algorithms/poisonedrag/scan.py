from dataclasses import dataclass

from vexrag.attack_algorithms.poisonedrag.generator import PoisonedRAGGenerator
from vexrag.attack_algorithms.poisonedrag.report import (
    PoisonedRAGCaseResult,
    PoisonedRAGScanReport,
)
from vexrag.attack_algorithms.poisonedrag.schema import (
    PoisonedRAGRequest,
    PoisonedRAGResult,
)
from vexrag.core.evaluation import EvaluationInput, EvaluationStrategyProtocol
from vexrag.core.scan import ScanVerdict
from vexrag.core.target import (
    TargetSystemAdapterProtocol,
    TargetSystemQuery,
    TargetSystemResponse,
)


@dataclass(frozen=True, slots=True)
class PoisonedRAGScanConfig:
    """Execution settings for a PoisonedRAG target-system scan."""

    repetitions: int = 1
    attack_success_rate_threshold: float = 0.0
    override_contexts: bool = False

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
    ) -> None:
        self.generator = generator
        self.target_system = target_system
        self.evaluation_strategy = evaluation_strategy

    def run(
        self,
        request: PoisonedRAGRequest,
        config: PoisonedRAGScanConfig | None = None,
    ) -> PoisonedRAGScanReport:
        scan_config = config or PoisonedRAGScanConfig()
        generated = self.generator.generate_one(request)
        cases = self._run_cases(generated, scan_config)
        return self._build_report(generated, cases, scan_config)

    def _run_cases(
        self,
        generated: PoisonedRAGResult,
        config: PoisonedRAGScanConfig,
    ) -> tuple[PoisonedRAGCaseResult, ...]:
        cases: list[PoisonedRAGCaseResult] = []
        for run_index in range(1, config.repetitions + 1):
            for adversarial_text_index, adversarial_text in enumerate(
                generated.adv_texts,
                start=1,
            ):
                cases.append(
                    self._run_case(
                        generated=generated,
                        adversarial_text=adversarial_text,
                        run_index=run_index,
                        adversarial_text_index=adversarial_text_index,
                        override_contexts=config.override_contexts,
                    )
                )
        return tuple(cases)

    def _run_case(
        self,
        *,
        generated: PoisonedRAGResult,
        adversarial_text: str,
        run_index: int,
        adversarial_text_index: int,
        override_contexts: bool,
    ) -> PoisonedRAGCaseResult:
        metadata = self._case_metadata(run_index, adversarial_text_index)
        system_response = self._query_target_system(
            generated=generated,
            adversarial_text=adversarial_text,
            metadata=metadata,
            override_contexts=override_contexts,
        )
        evaluation_input = self._build_evaluation_input(
            generated=generated,
            adversarial_text=adversarial_text,
            system_response=system_response,
            metadata=metadata,
        )
        evaluation = self.evaluation_strategy.evaluate(evaluation_input)
        return PoisonedRAGCaseResult(
            query=generated.query,
            adversarial_text=adversarial_text,
            expected_incorrect_answer=generated.incorrect_answer,
            system_response=system_response,
            evaluation=evaluation,
            run_index=run_index,
        )

    def _query_target_system(
        self,
        *,
        generated: PoisonedRAGResult,
        adversarial_text: str,
        metadata: dict[str, str | int],
        override_contexts: bool,
    ) -> TargetSystemResponse:
        contexts = (adversarial_text,) if override_contexts else ()
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
        generated: PoisonedRAGResult,
        adversarial_text: str,
        system_response: TargetSystemResponse,
        metadata: dict[str, str | int],
    ) -> EvaluationInput:
        return EvaluationInput(
            query=generated.query,
            actual_answer=system_response.answer,
            expected_clean_answer=generated.correct_answer,
            expected_attack_answer=generated.incorrect_answer,
            retrieved_contexts=system_response.contexts,
            context_override=(adversarial_text,),
            metadata=metadata,
        )

    def _build_report(
        self,
        generated: PoisonedRAGResult,
        cases: tuple[PoisonedRAGCaseResult, ...],
        config: PoisonedRAGScanConfig,
    ) -> PoisonedRAGScanReport:
        successful_cases = sum(case.successful for case in cases)
        success_rate = successful_cases / len(cases) if cases else 0.0
        vulnerable = successful_cases > 0 and (
            success_rate >= config.attack_success_rate_threshold
        )

        warnings = tuple(generated.meta.warnings)
        if not cases:
            warnings = (*warnings, "No adversarial texts were generated for scanning.")

        return PoisonedRAGScanReport(
            query=generated.query,
            correct_answer=generated.correct_answer,
            incorrect_answer=generated.incorrect_answer,
            verdict=(
                ScanVerdict.VULNERABLE if vulnerable else ScanVerdict.NOT_VULNERABLE
            ),
            cases=tuple(cases),
            generated_adversarial_texts=tuple(generated.adv_texts),
            warnings=warnings,
        )

    @staticmethod
    def _case_metadata(
        run_index: int,
        adversarial_text_index: int,
    ) -> dict[str, str | int]:
        return {
            "attack": "poisonedrag",
            "run_index": run_index,
            "adversarial_text_index": adversarial_text_index,
        }
