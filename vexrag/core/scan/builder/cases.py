import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vexrag.core.scan.config.errors import ScanConfigError


def path_strings_from_value(value: Any, prefix: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise ScanConfigError(f"{prefix} must be a string or a list of strings")

    paths: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ScanConfigError(f"{prefix}[{index}] must be a non-empty string")
        paths.append(item.strip())
    return tuple(paths)


def case_configs_from_value(value: Any, prefix: str) -> tuple[Mapping[str, Any], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScanConfigError(f"{prefix} must be a list of case mappings")

    cases: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ScanConfigError(f"{prefix}[{index}] must be a mapping")
        cases.append(item)
    return tuple(cases)


def load_case_file(raw_cases: str, *, path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw_cases.splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(raw_cases)

    try:
        import yaml
    except ImportError as exc:
        raise ScanConfigError("PyYAML is required to read YAML cases files") from exc
    try:
        return yaml.safe_load(raw_cases)
    except yaml.YAMLError as exc:
        raise ValueError("cases file is invalid YAML") from exc


def load_case_configs(
    file_path: str,
    *,
    base_dir: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    path = Path(file_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    try:
        raw_cases = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScanConfigError(f"could not read cases file: {path}") from exc

    try:
        loaded = load_case_file(raw_cases, path=path)
    except ValueError as exc:
        raise ScanConfigError(f"could not parse cases file: {path}") from exc

    if isinstance(loaded, Mapping) and "cases" in loaded:
        loaded = loaded["cases"]
    return case_configs_from_value(loaded, str(path))
