from vexrag.attack_algorithms.poison_base.profile import CorpusPoisonScanProfile

POISONEDRAG_ATTACK_ID = "poisonedrag"

POISONEDRAG_SCAN_PROFILE = CorpusPoisonScanProfile(
    attack_id=POISONEDRAG_ATTACK_ID,
    corpus_cleanup_label="poisoned texts",
    generate_log_verb="Generating",
    generated_log_verb="Generated",
    empty_requests_error="at least one PoisonedRAG case is required",
    default_adv_per_query=5,
)
