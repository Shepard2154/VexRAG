import pytest

from vexrag.core.attacks import (
    ensure_builtin_attacks_registered,
    reset_builtin_registration_for_tests,
)
from vexrag.core.attacks.registry import reset_default_attack_registry_for_tests


@pytest.fixture(autouse=True)
def _fresh_default_registry() -> None:
    reset_default_attack_registry_for_tests()
    reset_builtin_registration_for_tests()
    ensure_builtin_attacks_registered()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that exercise broader wiring (typically no network)",
    )
