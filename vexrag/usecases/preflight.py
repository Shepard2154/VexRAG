import json
import socket
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError

_ollama_models_cache: dict[str, set[str]] = {}
_vllm_models_cache: dict[str, set[str]] = {}


def clear_ollama_models_cache() -> None:
    _ollama_models_cache.clear()


def clear_vllm_models_cache() -> None:
    _vllm_models_cache.clear()


def preflight_target_system(config: Mapping[str, Any]) -> None:
    target_http = _target_http_config(config)
    if not target_http:
        return
    base_url = target_http.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        return
    parsed = urlparse(base_url.strip())
    host = parsed.hostname
    port = parsed.port
    if host is None:
        return
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        route = str(target_http.get("route", "")).strip()
        hint = f"{base_url.rstrip('/')}/{route.lstrip('/')}" if route else base_url
        raise UseCaseConfigError(
            "target_system is unreachable. "
            f"Could not connect to {host}:{port} ({exc}). "
            f"Start your RAG service first (configured endpoint: {hint})."
        ) from exc


def preflight_ollama_models(config: Mapping[str, Any]) -> None:
    required_models = collect_ollama_models(config)
    for base_url, models in required_models.items():
        if not models:
            continue
        available = fetch_ollama_models(base_url)
        missing = sorted(
            model for model in models if not _model_available(model, available)
        )
        if missing:
            raise UseCaseDependencyError(
                "ollama model(s) not available at "
                f"{base_url}: {', '.join(missing)}. "
                "Pull them first (for example: ollama pull <model>)."
            )


def preflight_vllm_models(config: Mapping[str, Any]) -> None:
    required_models = collect_vllm_models(config)
    for base_url, models in required_models.items():
        if not models:
            continue
        available = fetch_vllm_models(base_url)
        missing = sorted(
            model for model in models if not _model_available(model, available)
        )
        if missing:
            raise UseCaseDependencyError(
                "vLLM model(s) not available at "
                f"{base_url}: {', '.join(missing)}. "
                "Update your config to use deployed models or deploy missing models first."
            )


def collect_ollama_models(config: Mapping[str, Any]) -> dict[str, set[str]]:
    return _collect_provider_models(config, provider_name="ollama")


def collect_vllm_models(config: Mapping[str, Any]) -> dict[str, set[str]]:
    return _collect_provider_models(config, provider_name="vllm")


def _collect_provider_models(
    config: Mapping[str, Any], *, provider_name: str
) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    def _visit(node: Any) -> None:
        if isinstance(node, Mapping):
            provider = node.get("provider")
            base_url = node.get("base_url")
            model = node.get("model")
            if (
                isinstance(provider, str)
                and provider.strip().lower() == provider_name
                and isinstance(base_url, str)
                and base_url.strip()
                and isinstance(model, str)
                and model.strip()
            ):
                found.setdefault(base_url.strip().rstrip("/"), set()).add(model.strip())
            for value in node.values():
                _visit(value)
            return
        if isinstance(node, list | tuple):
            for item in node:
                _visit(item)

    _visit(config)
    return found


def fetch_ollama_models(base_url: str) -> set[str]:
    normalized = base_url.strip().rstrip("/")
    if normalized in _ollama_models_cache:
        return _ollama_models_cache[normalized]

    request = Request(
        f"{normalized}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError) as exc:
        raise UseCaseDependencyError(
            "could not query Ollama model list at "
            f"{normalized}/api/tags ({exc}). "
            "Ensure Ollama is running and reachable."
        ) from exc
    models_raw = payload.get("models")
    if not isinstance(models_raw, list):
        raise UseCaseDependencyError(
            f"unexpected Ollama /api/tags response at {normalized}: missing models list"
        )
    names: set[str] = set()
    for model_info in models_raw:
        if not isinstance(model_info, Mapping):
            continue
        name = model_info.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    _ollama_models_cache[normalized] = names
    return names


def _vllm_models_list_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def fetch_vllm_models(base_url: str) -> set[str]:
    normalized = base_url.strip().rstrip("/")
    if normalized in _vllm_models_cache:
        return _vllm_models_cache[normalized]

    models_url = _vllm_models_list_url(normalized)
    request = Request(
        models_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError) as exc:
        raise UseCaseDependencyError(
            "could not query vLLM model list at "
            f"{models_url} ({exc}). "
            "Ensure vLLM is running and reachable."
        ) from exc
    models_raw = payload.get("data")
    if not isinstance(models_raw, list):
        raise UseCaseDependencyError(
            f"unexpected vLLM models response at {models_url}: missing data list"
        )
    names: set[str] = set()
    for model_info in models_raw:
        if not isinstance(model_info, Mapping):
            continue
        model_id = model_info.get("id")
        if isinstance(model_id, str) and model_id.strip():
            names.add(model_id.strip())
    _vllm_models_cache[normalized] = names
    return names


def _target_http_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    target = config.get("target_system")
    if not isinstance(target, Mapping):
        return None
    http_config = target.get("http", target)
    if not isinstance(http_config, Mapping):
        return None
    return http_config


def _model_available(required: str, available: set[str]) -> bool:
    if required in available:
        return True
    if ":" in required:
        return False
    prefix = f"{required}:"
    return any(candidate.startswith(prefix) for candidate in available)
