from pathlib import Path

from vexrag.attack_algorithms.registries import create_scan_registries
from vexrag.core.retrieval.poisoning.adapters.file_text import FileTextPoisoner
from vexrag.core.scan.builder import build_corpus_poisoner


class TestBuildCorpusPoisoner:
    def test_build_file_text_poisoner_from_config(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "contexts"
        cfg = {
            "scan": {
                "corpus_poisoning": {
                    "backend": "file_text",
                    "path": str(corpus_dir),
                    "filename_prefix": "smoke",
                    "cleanup": True,
                },
            },
        }
        poisoner = build_corpus_poisoner(cfg, registries=create_scan_registries())
        assert isinstance(poisoner, FileTextPoisoner)
        assert poisoner.path == corpus_dir
        assert poisoner.filename_prefix == "smoke"

    def test_build_file_text_poisoner_resolves_relative_path(
        self, tmp_path: Path
    ) -> None:
        cfg = {
            "scan": {
                "corpus_poisoning": {
                    "backend": "file_text",
                    "path": "./contexts",
                    "cleanup": True,
                },
            },
        }
        poisoner = build_corpus_poisoner(
            cfg,
            registries=create_scan_registries(base_dir=tmp_path),
        )
        assert isinstance(poisoner, FileTextPoisoner)
        assert poisoner.path == tmp_path / "contexts"

    def test_build_corpus_poisoner_returns_none_when_disabled(self) -> None:
        cfg = {"scan": {"corpus_poisoning": False}}
        assert build_corpus_poisoner(cfg, registries=create_scan_registries()) is None
