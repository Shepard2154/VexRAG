import json
from pathlib import Path

import pytest

from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.scan.builder import build_corpus_poisoner
from vexrag.core.scan.config.errors import ScanConfigError

_EMBEDDING_OLLAMA = {
    "provider": "ollama",
    "base_url": "http://127.0.0.1:1",
    "endpoint": "/api/embed",
    "model": "m",
}


def test_build_vector_poisoner_requires_embedding_client() -> None:
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
        build_corpus_poisoner(cfg, registries=create_scan_registries())


def test_build_chroma_poisoner_rejects_host_and_persist_path() -> None:
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
        build_corpus_poisoner(cfg, registries=create_scan_registries())


def test_build_qdrant_poisoner_rejects_url_and_path() -> None:
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
        build_corpus_poisoner(cfg, registries=create_scan_registries())


def test_faiss_poison_add_and_delete_roundtrip(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    import numpy as np

    from vexrag.core.retrieval import FaissPoisoner

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

    adapter = FaissPoisoner(d, StubEmb())
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
