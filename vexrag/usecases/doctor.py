from collections.abc import Callable, Mapping
from typing import Any

from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.preflight import (
    collect_vllm_models,
    preflight_ollama_models,
    preflight_target_system,
    preflight_vllm_models,
)
from vexrag.usecases.scan_service import build_scan_command
from vexrag.usecases.types import DoctorCheckResult, DoctorResult

_VLLM_CHECK_NAME = "vLLM endpoint + required models"


def run_doctor(
    config: Mapping[str, Any],
    *,
    base_dir: Any,
    check_llms: bool = False,
) -> DoctorResult:
    config_label = (
        "scan config validity + LLM probe" if check_llms else "scan config validity"
    )
    results = [
        _run_check("target API availability", lambda: preflight_target_system(config)),
        _run_check(
            "Ollama endpoint + required models",
            lambda: preflight_ollama_models(config),
        ),
        _check_vllm(config),
        _run_check(
            config_label,
            lambda: build_scan_command(
                config,
                base_dir=base_dir,
                probe_llms=check_llms,
            ),
        ),
    ]
    return DoctorResult(checks=tuple(results))


def _run_check(name: str, check: Callable[[], None]) -> DoctorCheckResult:
    try:
        check()
    except (
        ScanConfigError,
        UseCaseConfigError,
        UseCaseDependencyError,
        ValueError,
        RuntimeError,
    ) as exc:
        return DoctorCheckResult(name=name, ok=False, error=str(exc))
    return DoctorCheckResult(name=name, ok=True, error=None)


def _check_vllm(config: Mapping[str, Any]) -> DoctorCheckResult:
    if not collect_vllm_models(config):
        return DoctorCheckResult(
            name=_VLLM_CHECK_NAME, ok=True, error=None, skipped=True
        )
    return _run_check(_VLLM_CHECK_NAME, lambda: preflight_vllm_models(config))
