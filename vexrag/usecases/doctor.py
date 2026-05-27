from collections.abc import Callable, Mapping
from typing import Any

from vexrag.core.scan.config.errors import ScanConfigError
from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.preflight import preflight_ollama_models, preflight_target_system
from vexrag.usecases.scan_service import build_scan_command


def run_doctor(
    config: Mapping[str, Any],
    *,
    base_dir: Any,
    check_llms: bool = False,
) -> int:
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

    results: list[tuple[str, bool, str | None]] = []
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
            results.append((name, False, str(exc)))
        else:
            results.append((name, True, None))

    print("VexRAG Doctor")
    print()
    for name, ok, error in results:
        icon = "OK" if ok else "FAIL"
        print(f"[{icon}] {name}")
        if error:
            print(f"  -> {error}")

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"Doctor verdict: FAIL ({len(failed)} check(s) failed)")
        return 1
    print("Doctor verdict: PASS")
    return 0
