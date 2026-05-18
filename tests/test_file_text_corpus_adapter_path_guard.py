import pytest

from vexrag.core.retrieval.adapters.file_text import FileTextCorpusAdapter
from vexrag.core.retrieval.errors import RetrievalCorpusError


def test_file_text_corpus_adapter_rejects_delete_outside_root(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    outsider = tmp_path / "evil.txt"
    outsider.write_text("x", encoding="utf-8")

    adapter = FileTextCorpusAdapter(corpus, "pfx")
    with pytest.raises(RetrievalCorpusError, match="outside corpus"):
        adapter.delete_texts([str(outsider)])
