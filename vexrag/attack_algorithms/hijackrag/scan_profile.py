from typing import Any

from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile

HIJACKRAG_ATTACK_ID = "hijackrag"


def _hijack_metadata_extra(_request: Any, generated: Any) -> dict[str, list[str]]:
    return {"hijack_segment_ids": list(generated.meta.segment_ids)}


HIJACKRAG_SCAN_PROFILE = CorpusPoisonScanProfile(
    attack_id=HIJACKRAG_ATTACK_ID,
    corpus_cleanup_label="hijack texts",
    generate_log_verb="Building HijackRAG",
    generated_log_verb="Built",
    empty_requests_error="at least one HijackRAG case is required",
    default_adv_per_query=1,
    metadata_extra=_hijack_metadata_extra,
)
