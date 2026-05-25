from typing import Any

from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile
from vexrag.attack_algorithms.poison_base.runner import CorpusPoisonScanRunner
from vexrag.attack_algorithms.poison_base.scan_config import (
    CorpusPoisonScanConfig as HijackRAGScanConfig,
)
from vexrag.core.evaluation import Evaluator
from vexrag.core.retrieval import CorpusPoisoner
from vexrag.core.target_systems import TargetSystemAdapter


def _hijack_metadata_extra(_request: Any, generated: Any) -> dict[str, list[str]]:
    return {"hijack_segment_ids": list(generated.meta.segment_ids)}


HIJACKRAG_SCAN_PROFILE = CorpusPoisonScanProfile(
    attack_id="hijackrag",
    corpus_cleanup_label="hijack texts",
    generate_log_verb="Building HijackRAG",
    generated_log_verb="Built",
    empty_requests_error="at least one HijackRAG case is required",
    metadata_extra=_hijack_metadata_extra,
)


class HijackRAGScanRunner(CorpusPoisonScanRunner):
    def __init__(
        self,
        generator: object,
        target_system: TargetSystemAdapter,
        evaluator: Evaluator,
        corpus_poisoner: CorpusPoisoner | None = None,
    ) -> None:
        super().__init__(
            profile=HIJACKRAG_SCAN_PROFILE,
            generator=generator,
            target_system=target_system,
            evaluator=evaluator,
            corpus_poisoner=corpus_poisoner,
        )


__all__ = [
    "HIJACKRAG_SCAN_PROFILE",
    "HijackRAGScanConfig",
    "HijackRAGScanRunner",
]
