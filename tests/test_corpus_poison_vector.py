import json
from pathlib import Path

import pytest

from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.retrieval.errors import CorpusPoisoningError
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


def test_faiss_persist_partial_failure_detected_on_next_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    import numpy as np

    from vexrag.core.retrieval import FaissPoisoner

    class StubEmb:
        dim = 4

        def embed_texts(self, texts):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    d = tmp_path / "faiss_corpus_partial_write"
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
    from vexrag.core.retrieval.poisoning.adapters import faiss as faiss_adapter_module

    call_count = {"n": 0}
    failed_once = {"done": False}
    real_replace = faiss_adapter_module.os.replace

    def flaky_replace(src: str, dst: str) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2 and not failed_once["done"]:
            failed_once["done"] = True
            raise OSError("simulated second replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(faiss_adapter_module.os, "replace", flaky_replace)

    with pytest.raises(CorpusPoisoningError, match="could not persist faiss index"):
        adapter.add_texts(["poison passage"], {})

    ids = adapter.add_texts(["another poison"], {})
    assert len(ids) == 1
    index_after = faiss.read_index(str(d / "index.faiss"))
    assert index_after.ntotal == 3
    meta_after = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert len(meta_after["ordered_ids"]) == 3


def test_faiss_poison_reopen_readable_after_add_delete(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    import numpy as np

    from vexrag.core.retrieval import FaissPoisoner

    class StubEmb:
        dim = 4

        def embed_texts(self, texts):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    d = tmp_path / "faiss_corpus_reopen"
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
    adapter.delete_texts(ids)

    adapter_reopened = FaissPoisoner(d, StubEmb())
    ids_reopened = adapter_reopened.add_texts(["fresh poison"], {})
    assert len(ids_reopened) == 1
    adapter_reopened.delete_texts(ids_reopened)

    index_final = faiss.read_index(str(d / "index.faiss"))
    meta_final = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert index_final.ntotal == 2
    assert meta_final["ordered_ids"] == [1, 2]
