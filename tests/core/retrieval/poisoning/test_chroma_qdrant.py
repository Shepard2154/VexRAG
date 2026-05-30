import pytest

from vexrag.core.retrieval import ChromaPoisoner, QdrantPoisoner


class _StubEmb:
    dim = 4

    def embed_texts(self, texts):  # type: ignore[no-untyped-def]
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestChromaQdrantRoundtrip:
    def test_chroma_add_and_delete_roundtrip(self, tmp_path) -> None:
        pytest.importorskip("chromadb")
        poisoner = ChromaPoisoner(
            persist_directory=tmp_path / "chroma",
            host=None,
            port=8000,
            collection_name="smoke",
            embedding_client=_StubEmb(),
        )
        ids = poisoner.add_texts(["poison passage"], {"case_id": "c1"})
        assert len(ids) == 1
        assert poisoner._collection.count() >= 1  # noqa: SLF001

        poisoner.delete_texts(ids)
        assert poisoner._collection.count() == 0  # noqa: SLF001

    def test_qdrant_local_path_add_and_delete_roundtrip(self, tmp_path) -> None:
        pytest.importorskip("qdrant_client")

        path = tmp_path / "qdrant"
        poisoner = QdrantPoisoner(
            url=None,
            path=path,
            collection="smoke",
            embedding_client=_StubEmb(),
        )
        poisoner._client.create_collection(  # noqa: SLF001
            collection_name="smoke",
            vectors_config=poisoner._qmodels.VectorParams(  # noqa: SLF001
                size=4,
                distance=poisoner._qmodels.Distance.COSINE,  # noqa: SLF001
            ),
        )
        ids = poisoner.add_texts(["poison passage"], {"case_id": "c1"})
        assert len(ids) == 1
        assert poisoner._client.count(collection_name="smoke").count >= 1  # noqa: SLF001

        poisoner.delete_texts(ids)
        assert poisoner._client.count(collection_name="smoke").count == 0  # noqa: SLF001
