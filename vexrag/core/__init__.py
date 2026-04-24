from enum import StrEnum

__all__ = ["RetrievalBackend"]


class RetrievalBackend(StrEnum):
    """backend used to hold retrieved context."""

    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"
    FILE_TEXT = "file_text"

    def uses_named_collection(self) -> bool:
        """True when the backend uses a collection name."""
        return self is not RetrievalBackend.FILE_TEXT
