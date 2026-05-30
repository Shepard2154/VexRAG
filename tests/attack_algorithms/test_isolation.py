"""Architecture guard: attack algorithm packages must not import each other."""

import ast
from pathlib import Path

_SHARED_PACKAGES = frozenset({"poison_base"})


def _attack_subpackages(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "vexrag" / "attack_algorithms"
    out: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if (
            child.is_dir()
            and not child.name.startswith("_")
            and (child / "__init__.py").is_file()
        ):
            out[child.name] = child
    return out


def _imports_other_attack(
    tree: ast.AST, *, package_name: str, others: set[str]
) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            prefix = "vexrag.attack_algorithms."
            if mod.startswith(prefix):
                foreign = mod[len(prefix) :].split(".", 1)[0]
                if (
                    foreign != package_name
                    and foreign in others
                    and foreign not in _SHARED_PACKAGES
                ):
                    violations.append(f"from {mod}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                prefix = "vexrag.attack_algorithms."
                if name.startswith(prefix):
                    foreign = name[len(prefix) :].split(".", 1)[0]
                    if (
                        foreign != package_name
                        and foreign in others
                        and foreign not in _SHARED_PACKAGES
                    ):
                        violations.append(f"import {name}")
    return violations


class TestAttackIsolation:
    def test_attack_packages_have_no_cross_imports(self, repo_root: Path) -> None:
        packages = _attack_subpackages(repo_root)
        names = set(packages.keys())
        assert names >= {"hijackrag", "poisonedrag"}

        for pkg_name, pkg_path in packages.items():
            for py_file in sorted(pkg_path.glob("*.py")):
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                bad = _imports_other_attack(tree, package_name=pkg_name, others=names)
                assert not bad, (
                    f"{py_file.relative_to(repo_root)} must not import other attacks: {bad}"
                )
