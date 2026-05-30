import pytest

from vexrag.attack_algorithms.poison_base.scan_config import (
    build_corpus_poison_scan_config,
)
from vexrag.core.scan.builder.options import (
    cleanup_option,
    correct_answer_style_option,
    poisoning_style_option,
    target_style_option,
)
from vexrag.core.scan.config.errors import ScanConfigError


class TestScanBuilderOptions:
    def test_target_style_option_accepts_short_fact(self) -> None:
        assert target_style_option({"target_style": "short_fact"}) == "short_fact"

    def test_target_style_option_rejects_unknown(self) -> None:
        with pytest.raises(ScanConfigError, match="target_style"):
            target_style_option({"target_style": "essay"})

    def test_correct_answer_style_prefers_dedicated_key(self) -> None:
        cfg = {
            "correct_answer_style": "paragraph",
            "target_style": "short_fact",
        }
        assert correct_answer_style_option(cfg) == "paragraph"

    @pytest.mark.parametrize("style", ["original", "aggressive", "soft"])
    def test_poisoning_style_option_accepts_known_values(self, style: str) -> None:
        assert poisoning_style_option({"poisoning_style": style}) == style

    def test_poisoning_style_option_rejects_unknown(self) -> None:
        with pytest.raises(ScanConfigError, match="poisoning_style"):
            poisoning_style_option({"poisoning_style": "extreme"})

    def test_cleanup_option_defaults_true_without_poison_section(self) -> None:
        assert cleanup_option({}) is True

    def test_cleanup_option_reads_scan_corpus_poisoning(self) -> None:
        cfg = {
            "corpus_poisoning": {
                "backend": "file_text",
                "cleanup": False,
            },
        }
        assert cleanup_option(cfg) is False


class TestCorpusPoisonScanConfig:
    def test_build_scan_config_reads_repetitions_and_threshold(self) -> None:
        cfg = {
            "scan": {
                "repetitions": 3,
                "attack_success_rate_threshold": 0.75,
                "override_contexts": True,
                "corpus_poisoning": {
                    "backend": "file_text",
                    "path": "./contexts",
                    "cleanup": False,
                },
            },
        }
        built = build_corpus_poison_scan_config(cfg)
        assert built.repetitions == 3
        assert built.attack_success_rate_threshold == 0.75
        assert built.override_contexts is True
        assert built.cleanup is False
