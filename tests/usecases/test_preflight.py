import pytest

from vexrag.usecases.errors import UseCaseConfigError, UseCaseDependencyError
from vexrag.usecases.preflight import (
    _vllm_models_list_url,
    clear_ollama_models_cache,
    collect_vllm_models,
    fetch_ollama_models,
    preflight_ollama_models,
    preflight_target_system,
)


class TestVLLMModelsListUrl:
    def test_vllm_models_list_url_when_base_includes_v1_suffix(self) -> None:
        assert (
            _vllm_models_list_url("http://localhost:8017/v1")
            == "http://localhost:8017/v1/models"
        )

    def test_vllm_models_list_url_when_base_is_host_root(self) -> None:
        assert (
            _vllm_models_list_url("http://localhost:8017")
            == "http://localhost:8017/v1/models"
        )


class TestCollectVLLMModels:
    def test_collect_vllm_models_empty_without_provider(self) -> None:
        assert collect_vllm_models({}) == {}

    def test_collect_vllm_models_finds_nested_provider(self) -> None:
        config = {
            "evaluation": {
                "provider": "vllm",
                "base_url": "http://127.0.0.1:8000",
                "model": "meta-llama/Llama-3-8B",
            },
        }
        found = collect_vllm_models(config)
        assert found == {"http://127.0.0.1:8000": {"meta-llama/Llama-3-8B"}}


class TestPreflightTarget:
    def test_preflight_target_system_passes_when_socket_connects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeSocket:
            def __enter__(self) -> "_FakeSocket":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(
            "vexrag.usecases.preflight.socket.create_connection",
            lambda *_a, **_k: _FakeSocket(),
        )
        preflight_target_system(
            {
                "target_system": {
                    "http": {"base_url": "http://localhost:8080"},
                },
            },
        )

    def test_preflight_target_system_fails_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise OSError("connection refused")

        monkeypatch.setattr(
            "vexrag.usecases.preflight.socket.create_connection",
            _fail,
        )
        with pytest.raises(UseCaseConfigError, match="target_system is unreachable"):
            preflight_target_system(
                {
                    "target_system": {
                        "http": {
                            "base_url": "http://localhost:8080",
                            "route": "/health",
                        },
                    },
                },
            )


class TestPreflightOllama:
    def test_preflight_ollama_models_passes_when_model_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_ollama_models_cache()
        monkeypatch.setattr(
            "vexrag.usecases.preflight.fetch_ollama_models",
            lambda _url: {"llama3:8b", "nomic-embed-text:latest"},
        )
        preflight_ollama_models(
            {
                "evaluation": {
                    "judge_client": {
                        "provider": "ollama",
                        "base_url": "http://localhost:11434",
                        "model": "llama3:8b",
                    },
                },
            },
        )

    def test_preflight_ollama_models_fails_when_model_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_ollama_models_cache()
        monkeypatch.setattr(
            "vexrag.usecases.preflight.fetch_ollama_models",
            lambda _url: {"nomic-embed-text:latest"},
        )
        with pytest.raises(UseCaseDependencyError, match="llama3:8b"):
            preflight_ollama_models(
                {
                    "evaluation": {
                        "judge_client": {
                            "provider": "ollama",
                            "base_url": "http://localhost:11434",
                            "model": "llama3:8b",
                        },
                    },
                },
            )

    def test_fetch_ollama_models_parses_tags_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_ollama_models_cache()

        class _Resp:
            def read(self) -> bytes:
                return b'{"models":[{"name":"llama3:8b"},{"name":"nomic-embed-text:latest"}]}'

            def __enter__(self) -> "_Resp":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(
            "vexrag.usecases.preflight.urlopen",
            lambda *_a, **_k: _Resp(),
        )
        models = fetch_ollama_models("http://localhost:11434")
        assert "llama3:8b" in models
