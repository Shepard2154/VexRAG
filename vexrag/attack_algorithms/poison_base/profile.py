from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

MetadataExtra = Callable[[Any, Any], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class CorpusPoisonScanProfile:
    attack_id: str
    corpus_cleanup_label: str
    generate_log_verb: str
    generated_log_verb: str
    empty_requests_error: str
    default_adv_per_query: int
    metadata_extra: MetadataExtra | None = None
