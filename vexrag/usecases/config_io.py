from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vexrag.usecases.errors import UseCaseConfigError


def load_config(path: Path) -> Mapping[str, Any]:
    try:
        raw_config = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UseCaseConfigError(f"could not read config file: {path}") from exc

    try:
        loaded = load_yaml(raw_config)
    except ValueError as exc:
        raise UseCaseConfigError(f"could not parse config file: {path}") from exc

    if not isinstance(loaded, Mapping):
        raise UseCaseConfigError("config file must contain a mapping")
    return loaded


def load_yaml(raw_config: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise UseCaseConfigError(
            "PyYAML is required to read YAML config files"
        ) from exc
    try:
        return yaml.safe_load(raw_config)
    except yaml.YAMLError as exc:
        raise UseCaseConfigError("YAML config file is invalid") from exc


def dump_yaml(content: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise UseCaseConfigError("PyYAML is required to write YAML files") from exc
    return yaml.safe_dump(
        content,
        allow_unicode=True,
        sort_keys=False,
    )


def write_yaml(path: Path, content: Mapping[str, Any]) -> None:
    text = dump_yaml(content)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise UseCaseConfigError(f"could not write YAML output file: {path}") from exc
