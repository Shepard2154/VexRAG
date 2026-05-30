from collections.abc import Mapping, Sequence
from typing import Any

from vexrag.core.llm.providers.config import provider_accessor
from vexrag.core.llm.providers.errors import ProviderConfigError


class SentenceTransformersEmbeddingClient:
    """Local SentenceTransformer embeddings (matches HF model ids)."""

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderConfigError(
                "sentence_transformers embedding provider requires optional dependencies; "
                "install with: pip install 'vexrag[sentence-transformers]'"
            ) from exc

        self._model_name = model
        self._normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model, device=device or "cpu")

    @property
    def model(self) -> str:
        return self._model_name

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        return tuple(row.tolist() for row in vectors)


def build_embedding_client(
    config: Mapping[str, Any],
) -> SentenceTransformersEmbeddingClient:
    accessor = provider_accessor(config, prefix="embedding_client")
    device_raw = config.get("device")
    device = (
        device_raw.strip()
        if isinstance(device_raw, str) and device_raw.strip()
        else None
    )
    return SentenceTransformersEmbeddingClient(
        accessor.get_required_string("model"),
        device=device,
        normalize_embeddings=accessor.get_bool("normalize_embeddings", True),
    )
