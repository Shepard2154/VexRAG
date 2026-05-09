import pytest

from vexrag.core.retrieval.poisoning.adapters.file_text import FileTextPoisoner
from vexrag.core.retrieval.poisoning.contracts import CorpusPoisoningError


def test_file_text_poisoner_rejects_delete_outside_root(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    outsider = tmp_path / "evil.txt"
    outsider.write_text("x", encoding="utf-8")

    poison = FileTextPoisoner(corpus, "pfx")
    with pytest.raises(CorpusPoisoningError, match="outside corpus"):
        poison.delete_texts([str(outsider)])
