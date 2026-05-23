from collections.abc import Mapping

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.scan.builder.retrieval import corpus_poisoning_section
from vexrag.core.scan.config.errors import ScanConfigError


def target_style_option(
    config: Mapping[str, object],
    *,
    default: str = "short_fact",
    prefix: str = "attack",
) -> str:
    value = str(config.get("target_style", default)).strip()
    if value not in {"short_fact", "paragraph"}:
        raise ScanConfigError(
            f"{prefix}.target_style must be 'short_fact' or 'paragraph'"
        )
    return value


def correct_answer_style_option(
    config: Mapping[str, object],
    *,
    default: str = "short_fact",
    prefix: str = "attack",
) -> str:
    """Prefer YAML ``correct_answer_style``; fall back to legacy ``target_style``."""
    raw = config.get("correct_answer_style")
    if raw is None:
        raw = config.get("target_style", default)
    value = str(raw).strip()
    if value not in {"short_fact", "paragraph"}:
        raise ScanConfigError(
            f"{prefix}.correct_answer_style (or legacy target_style) must be "
            "'short_fact' or 'paragraph'"
        )
    return value


def poisoning_style_option(
    config: Mapping[str, object],
    *,
    default: str = "original",
    prefix: str = "attack.poisonedrag",
) -> str:
    value = str(config.get("poisoning_style", default)).strip().lower()
    if value not in {"original", "aggressive", "soft"}:
        raise ScanConfigError(
            f"{prefix}.poisoning_style must be 'original', 'aggressive' or 'soft'"
        )
    return value


def cleanup_option(scan_config: Mapping[str, object]) -> bool:
    poison_config = corpus_poisoning_section({"scan": scan_config})
    if poison_config is None:
        return False
    accessor = ConfigAccessor(
        poison_config,
        prefix="scan.corpus_poisoning",
        error_type=ScanConfigError,
    )
    return accessor.get_bool("cleanup", False)
