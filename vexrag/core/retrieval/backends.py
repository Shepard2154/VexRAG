from enum import StrEnum


class RetrievalBackend(StrEnum):
    """Target RAG corpus backend."""

    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"
    FILE_TEXT = "file_text"
