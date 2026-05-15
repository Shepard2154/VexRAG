from enum import StrEnum


class EvaluationStrategy(StrEnum):
    EMBEDDING_SIMILARITY = "embedding_similarity"
    LLM_JUDGE = "llm_judge"
