from pathlib import Path


class TestUsecasePolicy:
    def test_usecases_have_no_print_calls(self, repo_root: Path) -> None:
        usecases_dir = repo_root / "vexrag" / "usecases"
        offenders: list[str] = []
        for path in sorted(usecases_dir.rglob("*.py")):
            if "print(" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(usecases_dir.parent.parent)))
        assert offenders == []
