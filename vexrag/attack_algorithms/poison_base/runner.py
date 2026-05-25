import logging
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile
from vexrag.attack_algorithms.poison_base.report import (
    CorpusPoisonCaseResult,
    CorpusPoisonScanReport,
)
from vexrag.attack_algorithms.poison_base.scan_config import CorpusPoisonScanConfig
from vexrag.core.evaluation import Evaluator
from vexrag.core.retrieval import CorpusPoisoner
from vexrag.core.scan.contracts import ScanCaseReport
from vexrag.core.scan.execution import probe_with_poisoning_and_evaluation
from vexrag.core.scan.types import ScanVerdict
from vexrag.core.target_systems import TargetSystemAdapter


class CorpusPoisonScanRunner:
    """Orchestrates corpus-poison generation and target-system checks."""

    def __init__(
        self,
        *,
        profile: CorpusPoisonScanProfile,
        generator: Any,
        target_system: TargetSystemAdapter,
        evaluator: Evaluator,
        corpus_poisoner: CorpusPoisoner | None = None,
    ) -> None:
        self._profile = profile
        self._logger = logging.getLogger(f"vexrag.scan.{profile.attack_id}")
        self.generator = generator
        self.target_system = target_system
        self.evaluator = evaluator
        self.corpus_poisoner = corpus_poisoner

    def run(
        self,
        requests: Sequence[Any],
        config: CorpusPoisonScanConfig | None = None,
        *,
        on_case_complete: Callable[[ScanCaseReport], None] | None = None,
    ) -> CorpusPoisonScanReport:
        scan_config = config or CorpusPoisonScanConfig()
        if not requests:
            raise ValueError(self._profile.empty_requests_error)

        cases: list[CorpusPoisonCaseResult] = []
        generated_results: list[Any] = []
        for case_index, request in enumerate(requests, start=1):
            case_label = request.case_id or f"#{case_index}"
            self._logger.info(
                "%s adversarial texts for case %s",
                self._profile.generate_log_verb,
                case_label,
            )
            generated = self.generator.generate_one(request)
            self._logger.info(
                "%s %d adversarial text(s) for case %s",
                self._profile.generated_log_verb,
                len(generated.adv_texts),
                case_label,
            )
            generated_results.append(generated)
            cases.extend(
                self._run_cases(
                    generated,
                    request,
                    case_index,
                    scan_config,
                    on_case_complete=on_case_complete,
                )
            )

        return self._build_report(tuple(generated_results), tuple(cases), scan_config)

    def _run_cases(
        self,
        generated: Any,
        request: Any,
        case_index: int,
        config: CorpusPoisonScanConfig,
        *,
        on_case_complete: Callable[[ScanCaseReport], None] | None = None,
    ) -> tuple[CorpusPoisonCaseResult, ...]:
        cases: list[CorpusPoisonCaseResult] = []
        for run_index in range(1, config.repetitions + 1):
            self._logger.info(
                "Querying target for case %s, run %d, %s %d",
                request.case_id or f"#{case_index}",
                run_index,
                self._profile.corpus_cleanup_label,
                len(generated.adv_texts),
            )
            finished = self._run_case(
                generated=generated,
                request=request,
                adversarial_texts=tuple(generated.adv_texts),
                case_index=case_index,
                run_index=run_index,
                override_contexts=config.override_contexts,
                cleanup=config.cleanup,
            )
            cases.append(finished)
            if on_case_complete is not None:
                on_case_complete(finished)
        return tuple(cases)

    def _run_case(
        self,
        *,
        generated: Any,
        request: Any,
        adversarial_texts: tuple[str, ...],
        case_index: int,
        run_index: int,
        override_contexts: bool,
        cleanup: bool,
    ) -> CorpusPoisonCaseResult:
        metadata = self._case_metadata(
            request=request,
            generated=generated,
            case_index=case_index,
            run_index=run_index,
            adversarial_text_count=len(adversarial_texts),
        )
        started = perf_counter()
        system_response, evaluation = probe_with_poisoning_and_evaluation(
            query=generated.query,
            correct_answer=generated.correct_answer,
            incorrect_answer=generated.incorrect_answer,
            adversarial_texts=adversarial_texts,
            corpus_poisoner=self.corpus_poisoner,
            target_system=self.target_system,
            evaluator=self.evaluator,
            override_contexts=override_contexts,
            cleanup=cleanup,
            metadata=metadata,
            corpus_cleanup_label=self._profile.corpus_cleanup_label,
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        self._logger.info(
            "case_done attack=%s case_id=%s run_index=%s duration_ms=%s "
            "completed=%s attack_successful=%s",
            self._profile.attack_id,
            metadata.get("case_id", ""),
            run_index,
            elapsed_ms,
            evaluation.completed,
            evaluation.attack_successful,
        )
        return CorpusPoisonCaseResult(
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
        generated_results: tuple[Any, ...],
        cases: tuple[CorpusPoisonCaseResult, ...],
        config: CorpusPoisonScanConfig,
    ) -> CorpusPoisonScanReport:
        evaluated = sum(1 for case in cases if case.evaluation.completed)
        successful_cases = sum(case.successful for case in cases)
        success_rate = successful_cases / evaluated if evaluated else 0.0
        vulnerable = (
            evaluated > 0
            and successful_cases > 0
            and (success_rate >= config.attack_success_rate_threshold)
        )

        warnings = tuple(
            warning
            for generated in generated_results
            for warning in generated.meta.warnings
        )
        if not cases:
            warnings = (*warnings, "No adversarial texts were generated for scanning.")
        unevaluated = len(cases) - evaluated
        if unevaluated:
            warnings = (
                *warnings,
                (
                    f"{unevaluated} case run(s) had incomplete evaluation and are "
                    "excluded from the attack success rate denominator."
                ),
            )

        return CorpusPoisonScanReport(
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

    def _case_metadata(
        self,
        *,
        request: Any,
        generated: Any,
        case_index: int,
        run_index: int,
        adversarial_text_count: int,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "attack": self._profile.attack_id,
            "case_index": case_index,
            "run_index": run_index,
            "adversarial_text_count": adversarial_text_count,
            "query": generated.query,
            "expected_clean_answer": generated.correct_answer,
            "expected_attack_answer": generated.incorrect_answer,
        }
        if request.case_id:
            metadata["case_id"] = request.case_id
        if self._profile.metadata_extra is not None:
            metadata.update(self._profile.metadata_extra(request, generated))
        return metadata
