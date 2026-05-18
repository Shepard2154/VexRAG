class RetrievalCorpusError(RuntimeError):
    """Retrieval corpus I/O failure."""


class RetrievalCorpusDependencyError(RetrievalCorpusError):
    """Required retrieval backend dependency is unavailable."""


class RetrievalCorpusBackendError(RetrievalCorpusError):
    """Retrieval backend operation failed."""


class RetrievalCorpusPersistenceError(RetrievalCorpusError):
    """Retrieval corpus persistence operation failed."""
