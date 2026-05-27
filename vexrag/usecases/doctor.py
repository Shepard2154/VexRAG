from collections.abc import Callable, Mapping
from typing import Any

from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.preflight import preflight_ollama_models, preflight_target_system
from vexrag.usecases.scan_service import build_scan_command
from vexrag.usecases.types import DoctorCheckResult, DoctorResult


def run_doctor(
    config: Mapping[str, Any],
    *,
    base_dir: Any,
    check_llms: bool = False,
) -> DoctorResult:
    config_label = (
        "scan config validity + LLM probe" if check_llms else "scan config validity"
    )
    checks: list[tuple[str, Callable[[], None]]] = [
        ("target API availability", lambda: preflight_target_system(config)),
        ("Ollama endpoint + required models", lambda: preflight_ollama_models(config)),
        (
            config_label,
            lambda: build_scan_command(
                config,
                base_dir=base_dir,
                probe_llms=check_llms,
            ),
        ),
    ]

    results: list[DoctorCheckResult] = []
    for name, check in checks:
        try:
            check()
        except (
            ScanConfigError,
            UseCaseConfigError,
            UseCaseDependencyError,
            ValueError,
            RuntimeError,
        ) as exc:
            results.append(DoctorCheckResult(name=name, ok=False, error=str(exc)))
        else:
            results.append(DoctorCheckResult(name=name, ok=True, error=None))

    return DoctorResult(checks=tuple(results))
