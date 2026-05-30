from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.core.base_configuration import ConfigAccessor
from vexrag.core.llm.contracts import EmbeddingClient
from vexrag.core.llm.providers.registry import LLMProviderRegistry
from vexrag.core.retrieval import (
    ChromaPoisoner,
    CorpusPoisoner,
    FaissPoisoner,
    FileTextPoisoner,
    QdrantPoisoner,
    RetrievalBackend,
)
from vexrag.core.retrieval.registry import RetrievalBackendRegistry
from vexrag.core.scan.builder.registries import ScanRegistries
from vexrag.core.scan.config.errors import ScanConfigError


def create_default_retrieval_backend_registry(
    *,
    llm_providers: LLMProviderRegistry,
    base_dir: Path | None,
) -> RetrievalBackendRegistry:
    return RetrievalBackendRegistry(
        {
            RetrievalBackend.FILE_TEXT: lambda config: _build_file_text_poisoner(
                config,
                base_dir=base_dir,
            ),
            RetrievalBackend.QDRANT: lambda config: _build_vector_poisoner(
                config,
                backend=RetrievalBackend.QDRANT,
                llm_providers=llm_providers,
                base_dir=base_dir,
            ),
            RetrievalBackend.CHROMA: lambda config: _build_vector_poisoner(
                config,
                backend=RetrievalBackend.CHROMA,
                llm_providers=llm_providers,
                base_dir=base_dir,
            ),
            RetrievalBackend.FAISS: lambda config: _build_vector_poisoner(
                config,
                backend=RetrievalBackend.FAISS,
                llm_providers=llm_providers,
                base_dir=base_dir,
            ),
        }
    )


def build_corpus_poisoner(
    config: Mapping[str, Any],
    *,
    registries: ScanRegistries,
) -> CorpusPoisoner | None:
    poison_config = corpus_poisoning_section(config)
    if poison_config is None:
        return None
    backend = str(poison_config.get("backend", poison_config.get("type", "file_text")))
    return registries.retrieval_backends.get(backend)(poison_config)


def corpus_poisoning_section(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    scan_config = config.get("scan", {})
    if not isinstance(scan_config, Mapping):
        raise ScanConfigError("scan must be a mapping")

    poison_config = scan_config.get("corpus_poisoning", config.get("retrieval"))
    if poison_config in (None, False):
        return None
    if not isinstance(poison_config, Mapping):
        raise ScanConfigError("scan.corpus_poisoning must be a mapping")
    return poison_config


def _build_file_text_poisoner(
    poison_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> FileTextPoisoner:
    accessor = ConfigAccessor(
        poison_config,
        prefix="scan.corpus_poisoning",
        base_dir=base_dir,
        error_type=ScanConfigError,
    )
    prefix_raw = poison_config.get("filename_prefix", "poisonedrag")
    filename_prefix = (
        prefix_raw.strip()
        if isinstance(prefix_raw, str) and prefix_raw.strip()
        else "poisonedrag"
    )
    return FileTextPoisoner(
        path=accessor.get_path("path", "directory", "contexts_dir", "corpus_path"),
        filename_prefix=filename_prefix,
    )


def _build_vector_poisoner(
    poison_config: Mapping[str, Any],
    *,
    backend: RetrievalBackend,
    llm_providers: LLMProviderRegistry,
    base_dir: Path | None,
) -> CorpusPoisoner:
    backend_config = _backend_config(poison_config, backend)
    embedding_section = _corpus_poisoning_embedding_section(
        backend_config,
        f"scan.corpus_poisoning.{backend.value}",
    )
    embedding_client = llm_providers.build_embedding_client(embedding_section)
    accessor = ConfigAccessor(
        poison_config,
        prefix="scan.corpus_poisoning",
        base_dir=base_dir,
        error_type=ScanConfigError,
    )
    l2_normalize = accessor.get_bool("l2_normalize", False)
    if backend is RetrievalBackend.QDRANT:
        return _build_qdrant_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    if backend is RetrievalBackend.CHROMA:
        return _build_chroma_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    if backend is RetrievalBackend.FAISS:
        return _build_faiss_corpus_poisoner(
            backend_config,
            embedding_client=embedding_client,
            l2_normalize=l2_normalize,
            base_dir=base_dir,
        )
    raise ScanConfigError(f"unsupported corpus poisoning backend: {backend.value}")


def _corpus_poisoning_embedding_section(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> Mapping[str, Any]:
    emb = backend_cfg.get("embedding_client")
    if not isinstance(emb, Mapping):
        raise ScanConfigError(
            f"{prefix}.embedding_client must be configured "
            "(must match the target RAG embedding model and preprocessing)"
        )
    return emb


def _vector_backend_collection_name(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> str:
    for key in ("collection", "collection_name"):
        raw = backend_cfg.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise ScanConfigError(f"{prefix} must set collection or collection_name")


def _optional_poison_timeout(
    backend_cfg: Mapping[str, Any],
    prefix: str,
) -> float | None:
    value = backend_cfg.get("timeout")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScanConfigError(f"{prefix}.timeout must be a number or null")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ScanConfigError(f"{prefix}.timeout must be a number or null") from exc
    if out <= 0:
        raise ScanConfigError(f"{prefix}.timeout must be greater than 0")
    return out


def _build_qdrant_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> QdrantPoisoner:
    prefix = "scan.corpus_poisoning.qdrant"
    accessor = ConfigAccessor(backend_cfg, prefix=prefix, error_type=ScanConfigError)
    url = accessor.get_optional_string("url")
    path_str = backend_cfg.get("path")
    path: Path | None = None
    if isinstance(path_str, str) and path_str.strip():
        path = Path(path_str.strip())
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
    if url and path:
        raise ScanConfigError(f"{prefix} must set only one of url or path")
    if not url and not path:
        raise ScanConfigError(f"{prefix} must set url or path")
    return QdrantPoisoner(
        url=url,
        path=path,
        collection=_vector_backend_collection_name(backend_cfg, prefix),
        embedding_client=embedding_client,
        vector_name=accessor.get_optional_string("vector_name"),
        timeout=_optional_poison_timeout(backend_cfg, prefix),
        api_key=accessor.get_optional_string("api_key"),
        l2_normalize=l2_normalize,
    )


def _build_chroma_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> ChromaPoisoner:
    prefix = "scan.corpus_poisoning.chroma"
    accessor = ConfigAccessor(backend_cfg, prefix=prefix, error_type=ScanConfigError)
    host = accessor.get_optional_string("host")
    port = accessor.get_int("port", 8000)
    path_str = backend_cfg.get("persist_directory") or backend_cfg.get("path")
    persist_directory: Path | None = None
    if host:
        if isinstance(path_str, str) and path_str.strip():
            raise ScanConfigError(
                f"{prefix} cannot set persist_directory/path when host is set"
            )
    else:
        if not isinstance(path_str, str) or not path_str.strip():
            raise ScanConfigError(
                f"{prefix} must set persist_directory or path for local Chroma"
            )
        persist_directory = Path(path_str.strip())
        if not persist_directory.is_absolute() and base_dir is not None:
            persist_directory = base_dir / persist_directory
    return ChromaPoisoner(
        persist_directory=persist_directory,
        host=host,
        port=port,
        collection_name=_vector_backend_collection_name(backend_cfg, prefix),
        embedding_client=embedding_client,
        l2_normalize=l2_normalize,
    )


def _build_faiss_corpus_poisoner(
    backend_cfg: Mapping[str, Any],
    *,
    embedding_client: EmbeddingClient,
    l2_normalize: bool,
    base_dir: Path | None,
) -> FaissPoisoner:
    prefix = "scan.corpus_poisoning.faiss"
    accessor = ConfigAccessor(
        backend_cfg,
        prefix=prefix,
        base_dir=base_dir,
        error_type=ScanConfigError,
    )
    poison_start = backend_cfg.get("poison_id_start")
    poison_id_start = -1
    if poison_start is not None:
        if isinstance(poison_start, bool):
            raise ScanConfigError(f"{prefix}.poison_id_start must be an integer")
        try:
            poison_id_start = int(poison_start)
        except (TypeError, ValueError) as exc:
            raise ScanConfigError(
                f"{prefix}.poison_id_start must be an integer"
            ) from exc
    return FaissPoisoner(
        accessor.get_path("directory", "path", "faiss_dir"),
        embedding_client,
        l2_normalize=l2_normalize,
        poison_id_start=poison_id_start,
    )


def _backend_config(
    config: Mapping[str, Any],
    backend: RetrievalBackend,
) -> Mapping[str, Any]:
    nested = config.get(backend.value)
    if nested is None:
        return config
    if not isinstance(nested, Mapping):
        raise ScanConfigError(
            f"scan.corpus_poisoning.{backend.value} must be a mapping"
        )
    return {**config, **nested}
