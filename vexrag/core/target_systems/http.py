import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from vexrag.core.target_systems.errors import TargetSystemAdapterError
from vexrag.core.target_systems.types import (
    HTTPResponsePaths,
    HTTPTargetSystemAdapterConfig,
    TargetSystemQuery,
    TargetSystemResponse,
)


def _response_paths(config: HTTPTargetSystemAdapterConfig) -> HTTPResponsePaths:
    paths = config.response_paths
    assert isinstance(paths, HTTPResponsePaths)
    return paths


class HTTPTargetSystemAdapter:
    """Configurable HTTP implementation of TargetSystemAdapter."""

    def __init__(
        self,
        config: HTTPTargetSystemAdapterConfig,
        *,
        http_open: Callable[..., Any] = urlopen,
    ) -> None:
        self.config = config
        self._http_open = http_open

    def answer(self, request: TargetSystemQuery) -> TargetSystemResponse:
        payload = self._render_request_payload(request)
        response_payload, status_code = self._send(payload)
        answer = self._extract_answer(response_payload)
        contexts = self._extract_contexts(response_payload)
        return TargetSystemResponse(
            answer=answer,
            contexts=contexts,
            metadata=(
                {
                    "http_status": status_code,
                    "response_payload": response_payload,
                }
                if self.config.include_raw_response_in_metadata
                else {"http_status": status_code}
            ),
        )

    def _render_request_payload(self, request: TargetSystemQuery) -> Any:
        context = {
            "query": request.query,
            "contexts": list(request.contexts),
            "metadata": dict(request.metadata),
        }
        return self._render_template(self.config.request_template, context)

    def _send(self, payload: Any) -> tuple[Any, int | None]:
        method = self.config.method
        url = self._build_url()
        headers = {
            "Accept": "application/json",
            **self.config.headers,
        }
        data = None

        if method == "GET":
            url = self._url_with_query(url, payload)
        else:
            headers.setdefault("Content-Type", "application/json")
            data = json.dumps(payload).encode("utf-8")

        http_request = Request(url, data=data, headers=headers, method=method)
        try:
            with self._http_open(http_request, timeout=self.config.timeout) as response:
                raw_body = response.read()
                status_code = getattr(response, "status", None)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise TargetSystemAdapterError(
                f"target system returned HTTP {error.code}: {body}"
            ) from error
        except URLError as error:
            raise TargetSystemAdapterError(
                f"target system request failed: {error.reason}"
            ) from error

        try:
            return json.loads(raw_body.decode("utf-8")), status_code
        except json.JSONDecodeError as error:
            raise TargetSystemAdapterError(
                "target system returned a non-JSON response"
            ) from error

    def _extract_answer(self, payload: Any) -> str:
        value = self._extract_path(payload, _response_paths(self.config).answer)
        if value is None:
            raise TargetSystemAdapterError("response answer path resolved to null")
        return str(value)

    def _extract_contexts(self, payload: Any) -> tuple[str, ...]:
        path = _response_paths(self.config).contexts
        if path is None:
            return ()
        value = self._extract_path(payload, path)
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list | tuple):
            return tuple(str(item) for item in value)
        raise TargetSystemAdapterError(
            "response contexts path must resolve to a string, list, or null"
        )

    def _build_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        route = self.config.route.strip("/")
        return f"{base_url}/{route}" if route else base_url

    @staticmethod
    def _url_with_query(url: str, payload: Any) -> str:
        if not isinstance(payload, Mapping):
            raise TargetSystemAdapterError(
                "GET request_template must render to a mapping"
            )
        query_params = {
            key: HTTPTargetSystemAdapter._to_query_value(value)
            for key, value in payload.items()
        }
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{urlencode(query_params)}"

    @staticmethod
    def _to_query_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value)

    @staticmethod
    def _render_template(template: Any, context: Mapping[str, Any]) -> Any:
        if isinstance(template, Mapping):
            return {
                key: HTTPTargetSystemAdapter._render_template(value, context)
                for key, value in template.items()
            }
        if isinstance(template, list):
            return [
                HTTPTargetSystemAdapter._render_template(value, context)
                for value in template
            ]
        if isinstance(template, tuple):
            return tuple(
                HTTPTargetSystemAdapter._render_template(value, context)
                for value in template
            )
        if isinstance(template, str):
            return HTTPTargetSystemAdapter._render_string_template(template, context)
        return template

    @staticmethod
    def _render_string_template(template: str, context: Mapping[str, Any]) -> Any:
        fields = HTTPTargetSystemAdapter._template_fields(template)
        if len(fields) == 1 and template == f"{{{fields[0]}}}":
            return HTTPTargetSystemAdapter._extract_path(context, fields[0])

        rendered = template
        for template_field in fields:
            value = HTTPTargetSystemAdapter._extract_path(context, template_field)
            rendered = rendered.replace(f"{{{template_field}}}", str(value))
        return rendered

    @staticmethod
    def _template_fields(template: str) -> tuple[str, ...]:
        fields: list[str] = []
        start = 0
        while True:
            open_index = template.find("{", start)
            if open_index == -1:
                break
            close_index = template.find("}", open_index + 1)
            if close_index == -1:
                break
            field = template[open_index + 1 : close_index]
            if field:
                fields.append(field)
            start = close_index + 1
        return tuple(fields)

    @staticmethod
    def _extract_path(payload: Any, path: str) -> Any:
        if path in {"", "$"}:
            return payload

        current = payload
        for segment in path.split("."):
            if isinstance(current, Mapping):
                try:
                    current = current[segment]
                except KeyError as error:
                    raise TargetSystemAdapterError(
                        f"path '{path}' is missing segment '{segment}'"
                    ) from error
                continue

            if isinstance(current, list | tuple):
                try:
                    current = current[int(segment)]
                except (ValueError, IndexError) as error:
                    raise TargetSystemAdapterError(
                        f"path '{path}' is missing list index '{segment}'"
                    ) from error
                continue

            raise TargetSystemAdapterError(
                f"path '{path}' cannot traverse segment '{segment}'"
            )

        return current
