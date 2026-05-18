import logging
from typing import Any

from vexrag.core.evaluation import EvaluationInput, EvaluationResult, Evaluator
from vexrag.core.retrieval import RetrievalCorpusAdapter
from vexrag.core.target import (
    TargetSystemAdapterProtocol,
    TargetSystemQuery,
    TargetSystemResponse,
)

LOGGER = logging.getLogger("vexrag.adversarial_probe")


def probe_with_poisoning_and_evaluation(
    *,
    query: str,
    correct_answer: str,
    incorrect_answer: str,
    adversarial_texts: tuple[str, ...],
    corpus_adapter: RetrievalCorpusAdapter | None,
    target_system: TargetSystemAdapterProtocol,
    evaluator: Evaluator,
    override_contexts: bool,
    cleanup: bool,
    metadata: dict[str, Any],
    corpus_cleanup_label: str = "adversarial texts",
) -> tuple[TargetSystemResponse, EvaluationResult]:
    """Write adversarial texts to corpus (optional), query target, evaluate, optionally cleanup."""
    poisoned_document_ids = _poison_retrieval_corpus(
        corpus_adapter=corpus_adapter,
        adversarial_texts=adversarial_texts,
        metadata=metadata,
    )
    if poisoned_document_ids:
        metadata["poisoned_document_ids"] = poisoned_document_ids
    try:
        contexts = adversarial_texts if override_contexts else ()
        system_response = target_system.answer(
            TargetSystemQuery(
                query=query,
                contexts=contexts,
                metadata=metadata,
            )
        )
        evaluation_input = EvaluationInput(
            query=query,
            actual_answer=system_response.answer,
            expected_clean_answer=correct_answer,
            expected_attack_answer=incorrect_answer,
            retrieved_contexts=system_response.contexts,
            context_override=adversarial_texts if override_contexts else None,
            metadata=metadata,
        )
        evaluation = evaluator.evaluate(evaluation_input)
        return system_response, evaluation
    finally:
        _cleanup_poisoned_documents(
            corpus_adapter=corpus_adapter,
            poisoned_document_ids=poisoned_document_ids,
            cleanup=cleanup,
            label=corpus_cleanup_label,
        )


def _poison_retrieval_corpus(
    *,
    corpus_adapter: RetrievalCorpusAdapter | None,
    adversarial_texts: tuple[str, ...],
    metadata: dict[str, Any],
) -> tuple[str, ...]:
    if corpus_adapter is None or not adversarial_texts:
        return ()
    LOGGER.info(
        "Writing %d adversarial text(s) into retrieval corpus",
        len(adversarial_texts),
    )
    return corpus_adapter.add_texts(adversarial_texts, metadata)


def _cleanup_poisoned_documents(
    *,
    corpus_adapter: RetrievalCorpusAdapter | None,
    poisoned_document_ids: tuple[str, ...],
    cleanup: bool,
    label: str,
) -> None:
    if not cleanup or not poisoned_document_ids or corpus_adapter is None:
        return
    LOGGER.info("Cleaning up %s from retrieval corpus", label)
    corpus_adapter.delete_texts(poisoned_document_ids)
