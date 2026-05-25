from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile
from vexrag.attack_algorithms.poison_base.runner import CorpusPoisonScanRunner
from vexrag.attack_algorithms.poison_base.scan_config import (
    CorpusPoisonScanConfig as PoisonedRAGScanConfig,
)
from vexrag.core.evaluation import Evaluator
from vexrag.core.retrieval import CorpusPoisoner
from vexrag.core.target_systems import TargetSystemAdapter

POISONEDRAG_SCAN_PROFILE = CorpusPoisonScanProfile(
    attack_id="poisonedrag",
    corpus_cleanup_label="poisoned texts",
    generate_log_verb="Generating",
    generated_log_verb="Generated",
    empty_requests_error="at least one PoisonedRAG case is required",
)


class PoisonedRAGScanRunner(CorpusPoisonScanRunner):
    """Orchestrates PoisonedRAG generation and target-system checks."""

    def __init__(
        self,
        generator: object,
        target_system: TargetSystemAdapter,
        evaluator: Evaluator,
        corpus_poisoner: CorpusPoisoner | None = None,
    ) -> None:
        super().__init__(
            profile=POISONEDRAG_SCAN_PROFILE,
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=corpus_poisoner,
        )


__all__ = [
    "POISONEDRAG_SCAN_PROFILE",
    "PoisonedRAGScanConfig",
    "PoisonedRAGScanRunner",
]
