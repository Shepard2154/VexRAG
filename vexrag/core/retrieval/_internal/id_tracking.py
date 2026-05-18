from collections.abc import Iterable, MutableSet, Sequence


def collect_created_document_ids_for_cleanup(
    document_ids: Sequence[str],
    created_document_ids: set[str],
) -> tuple[str, ...]:
    """Return created document IDs that this adapter instance may clean up."""
    cleanup_ids: list[str] = []
    seen: set[str] = set()
    for raw_document_id in document_ids:
        if not isinstance(raw_document_id, str):
            continue
        document_id = raw_document_id.strip()
        if (
            document_id
            and document_id in created_document_ids
            and document_id not in seen
        ):
            cleanup_ids.append(document_id)
            seen.add(document_id)
    return tuple(cleanup_ids)


def remember_created_document_ids(
    created_document_ids: MutableSet[str],
    document_ids: Iterable[str],
) -> tuple[str, ...]:
    remembered = tuple(document_ids)
    created_document_ids.update(remembered)
    return remembered


def forget_created_document_ids(
    created_document_ids: MutableSet[str],
    document_ids: Iterable[str],
) -> None:
    for document_id in document_ids:
        created_document_ids.discard(document_id)
