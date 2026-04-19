"""Core types for retrieval backends."""

from enum import StrEnum

__all__ = ["RetrievalBackend"]


class RetrievalBackend(StrEnum):
    """backend used to hold retrieved context."""

    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"
    FILE_TEXT = "file_text"

    def uses_named_collection(self) -> bool:
        """True for Qdrant / FAISS / Chroma: config identifies a collection or index by name.

        False for ``file_text``: context is loaded from filesystem paths, not a named store.
        """
        return self is not RetrievalBackend.FILE_TEXT
