import json
import math
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vexrag.core.providers.errors import ProviderServiceError


def post_json(
    *,
    base_url: str,
    endpoint: str,
    payload: Mapping[str, Any],
    timeout: float | None,
    service_name: str,
    api_key: str | None = None,
) -> Mapping[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except HTTPError as error:
        body = _read_http_error_body(error)
        raise ProviderServiceError(
            f"{service_name} request returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise ProviderServiceError(
            f"{service_name} request failed: {error.reason}"
        ) from error
    except (OSError, ValueError) as error:
        raise ProviderServiceError(f"{service_name} request failed: {error}") from error

    return _decode_json_response(raw_body, service_name)


def _decode_json_response(raw_body: bytes, service_name: str) -> Mapping[str, Any]:
    try:
        decoded_body = raw_body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProviderServiceError(
            f"{service_name} response was not valid UTF-8"
        ) from error

    try:
        decoded = json.loads(decoded_body)
    except json.JSONDecodeError as error:
        raise ProviderServiceError(
            f"{service_name} response was not valid JSON"
        ) from error
    if not isinstance(decoded, Mapping):
        raise ProviderServiceError(f"{service_name} response must be a JSON object")
    return decoded


def _read_http_error_body(error: HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except OSError as body_error:
        return f"failed to read error body: {body_error}"


def coerce_embedding(vector: object) -> tuple[float, ...]:
    """Validate and coerce an embedding value parsed from untyped JSON."""
    if not isinstance(vector, list):
        raise ProviderServiceError("embedding response items must be numeric lists")

    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProviderServiceError("embedding response items must be numeric lists")
        coerced_value = float(value)
        if not math.isfinite(coerced_value):
            raise ProviderServiceError("embedding values must be finite numbers")
        values.append(coerced_value)
    return tuple(values)
