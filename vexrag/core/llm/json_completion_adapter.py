from collections.abc import Mapping
from typing import Any, cast


class JSONGenerationLLMClientAdapter:
    """Wraps a judge-style JSON LLM client for adversarial text generators."""

    def __init__(self, client: Any) -> None:
        self.client = client

    @property
    def model_id(self) -> str:
        return str(getattr(self.client, "model", getattr(self.client, "model_id", "")))

    def complete_json(self, prompt: str) -> str | Mapping[str, Any]:
        return cast(str | Mapping[str, Any], self.client.complete_json(prompt))
