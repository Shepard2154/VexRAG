import json
from pathlib import Path

import pytest

from vexrag.core.config import ScanConfigError
from vexrag.core.config.build import build_retrieval_corpus_adapter
from vexrag.core.retrieval import RetrievalBackend

_EMBEDDING_OLLAMA = {
    "provider": "ollama",
    "base_url": "http://127.0.0.1:1",
    "endpoint": "/api/embed",
    "model": "m",
}


def test_retrieval_backend_only_names_backend_type() -> None:
    assert not hasattr(RetrievalBackend.QDRANT, "uses_named_collection")


def test_build_vector_corpus_adapter_requires_embedding_client() -> None:
    cfg = {
        "scan": {
            "corpus_poisoning": {
                "backend": "qdrant",
                "path": "/tmp/vexrag-qdrant-test",
                "collection": "c",
            },
        },
    }
    with pytest.raises(ScanConfigError, match="embedding_client"):
        build_retrieval_corpus_adapter(cfg)


def test_build_chroma_corpus_adapter_rejects_host_and_persist_path() -> None:
    cfg = {
        "scan": {
            "corpus_poisoning": {
                "backend": "chroma",
                "host": "localhost",
                "path": "/tmp/chroma-data",
                "collection": "col",
                "embedding_client": _EMBEDDING_OLLAMA,
            },
        },
    }
    with pytest.raises(ScanConfigError, match="persist_directory"):
        build_retrieval_corpus_adapter(cfg)


def test_build_qdrant_corpus_adapter_rejects_url_and_path() -> None:
    cfg = {
        "scan": {
            "corpus_poisoning": {
                "backend": "qdrant",
                "url": "http://localhost:6333",
                "path": "/tmp/qdrant",
                "collection": "c",
                "embedding_client": _EMBEDDING_OLLAMA,
            },
        },
    }
    with pytest.raises(ScanConfigError, match="only one of url or path"):
        build_retrieval_corpus_adapter(cfg)


def test_faiss_corpus_adapter_add_and_delete_roundtrip(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    import numpy as np

    from vexrag.core.retrieval import FaissCorpusAdapter

    class StubEmb:
        dim = 4

        def embed_texts(self, texts):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    d = tmp_path / "faiss_corpus"
    d.mkdir()
    index = faiss.IndexFlatIP(4)
    vecs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    index.add(vecs)
    faiss.write_index(index, str(d / "index.faiss"))
    (d / "metadata.json").write_text(
        json.dumps({"ordered_ids": [1, 2]}),
        encoding="utf-8",
    )

    adapter = FaissCorpusAdapter(d, StubEmb())
    ids = adapter.add_texts(["poison passage"], {})
    assert len(ids) == 1

    index_after = faiss.read_index(str(d / "index.faiss"))
    assert index_after.ntotal == 3
    meta_after = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert len(meta_after["ordered_ids"]) == 3

    adapter.delete_texts(ids)
    index_final = faiss.read_index(str(d / "index.faiss"))
    assert index_final.ntotal == 2
    meta_final = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta_final["ordered_ids"] == [1, 2]


def test_faiss_poison_ids_do_not_repeat_across_adapter_instances(
    tmp_path: Path,
) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")

    from vexrag.core.retrieval import FaissCorpusAdapter

    class StubEmb:
        def embed_texts(self, texts):
            return [[0.5, 0.5] for _ in texts]

    d = tmp_path / "faiss_corpus"
    d.mkdir()
    index = faiss.IndexFlatIP(2)
    faiss.write_index(index, str(d / "index.faiss"))
    (d / "metadata.json").write_text(
        json.dumps({"ordered_ids": []}),
        encoding="utf-8",
    )

    first_adapter = FaissCorpusAdapter(d, StubEmb())
    second_adapter = FaissCorpusAdapter(d, StubEmb())

    first_ids = first_adapter.add_texts(["first poison"], {})
    second_ids = second_adapter.add_texts(["second poison"], {})

    assert first_ids == ("-1",)
    assert second_ids == ("-2",)
    meta_after = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta_after["ordered_ids"] == [-1, -2]


def test_faiss_delete_rejects_unsupported_index_without_changing_files(
    tmp_path: Path,
) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    import numpy as np

    from vexrag.core.retrieval import FaissCorpusAdapter, RetrievalCorpusError

    class StubEmb:
        def embed_texts(self, texts):
            return [[0.5, 0.5] for _ in texts]

    d = tmp_path / "faiss_corpus"
    d.mkdir()
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(d / "index.faiss"))
    metadata_path = d / "metadata.json"
    metadata_path.write_text(
        json.dumps({"ordered_ids": [1, -1]}),
        encoding="utf-8",
    )
    index_before = (d / "index.faiss").read_bytes()
    metadata_before = metadata_path.read_text(encoding="utf-8")

    adapter = FaissCorpusAdapter(d, StubEmb())
    adapter._created_document_ids.add("-1")

    with pytest.raises(RetrievalCorpusError, match="only supports IndexFlatIP"):
        adapter.delete_texts(["-1"])

    assert (d / "index.faiss").read_bytes() == index_before
    assert metadata_path.read_text(encoding="utf-8") == metadata_before


def test_faiss_persist_rolls_back_when_metadata_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vexrag.core.retrieval.adapters.faiss import _persist_faiss_corpus
    from vexrag.core.retrieval.errors import RetrievalCorpusPersistenceError

    class FakeFaiss:
        def write_index(self, index, path):
            Path(path).write_text("new-index", encoding="utf-8")

    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.json"
    index_path.write_text("old-index", encoding="utf-8")
    metadata_path.write_text("old-metadata", encoding="utf-8")

    real_replace = Path.replace

    def fail_metadata_install(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self.name.startswith(".metadata.json.tmp-") and target_path == metadata_path:
            raise OSError("simulated replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_install)

    with pytest.raises(RetrievalCorpusPersistenceError):
        _persist_faiss_corpus(
            index_path,
            metadata_path,
            object(),
            [1, -1],
            FakeFaiss(),
            os_error_message="could not persist faiss index",
        )

    assert index_path.read_text(encoding="utf-8") == "old-index"
    assert metadata_path.read_text(encoding="utf-8") == "old-metadata"
