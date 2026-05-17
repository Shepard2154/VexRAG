import json
from collections.abc import Mapping, Sequence
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

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ProviderServiceError(
            f"{service_name} request returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise ProviderServiceError(
            f"{service_name} request failed: {error.reason}"
        ) from error

    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderServiceError(
            f"{service_name} response was not valid JSON"
        ) from error
    if not isinstance(decoded, Mapping):
        raise ProviderServiceError(f"{service_name} response must be a JSON object")
    return decoded


def coerce_embedding(vector: object) -> tuple[float, ...]:
    if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
        raise ProviderServiceError("embedding response items must be numeric lists")
    try:
        return tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ProviderServiceError("embedding values must be numeric") from exc
