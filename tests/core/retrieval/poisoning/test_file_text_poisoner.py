from pathlib import Path

import pytest

from vexrag.core.retrieval.errors import CorpusPoisoningError
from vexrag.core.retrieval.poisoning.adapters.file_text import FileTextPoisoner


class TestFileTextPoisoner:
    def test_add_texts_writes_files_and_returns_ids(self, tmp_path) -> None:
        corpus = tmp_path / "corpus"
        poison = FileTextPoisoner(corpus, "pfx")
        ids = poison.add_texts(
            ["  poison passage  ", "second"],
            {"case_id": "c1", "run_index": 1},
        )
        assert len(ids) == 2
        for doc_id in ids:
            assert Path(doc_id).is_file()
            assert Path(doc_id).read_text(encoding="utf-8").endswith("\n")

    def test_delete_texts_removes_files(self, tmp_path) -> None:
        corpus = tmp_path / "corpus"
        poison = FileTextPoisoner(corpus, "pfx")
        ids = poison.add_texts(["to delete"], {"case_id": "c1"})
        assert all(Path(doc_id).is_file() for doc_id in ids)
        poison.delete_texts(ids)
        assert all(not Path(doc_id).exists() for doc_id in ids)

    def test_add_texts_skips_empty_strings(self, tmp_path) -> None:
        corpus = tmp_path / "corpus"
        poison = FileTextPoisoner(corpus, "pfx")
        ids = poison.add_texts(["", "   ", "\n"], {})
        assert ids == ()
        assert not any(corpus.iterdir()) if corpus.exists() else True


class TestFileTextPoisonerPathGuard:
    def test_file_text_poisoner_rejects_delete_outside_root(self, tmp_path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()

        outsider = tmp_path / "evil.txt"
        outsider.write_text("x", encoding="utf-8")

        poison = FileTextPoisoner(corpus, "pfx")
        with pytest.raises(CorpusPoisoningError, match="outside corpus"):
            poison.delete_texts([str(outsider)])
