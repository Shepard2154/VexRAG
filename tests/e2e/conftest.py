"""E2E fixtures: Ollama, RAG target subprocess, vx scan helper."""

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"
TARGET_HOST = "localhost"
TARGET_PORT = 8080
TARGET_HEALTH_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/health"
READINESS_TIMEOUT_S = 180
SCAN_TIMEOUT_S = 600
QDRANT_DOCKER_IMAGE = "qdrant/qdrant"
QDRANT_HOST_PORT = 16333
QDRANT_SERVER_READY_TIMEOUT_S = 30
QDRANT_SCAN_CONFIG_PLACEHOLDER_URL = "http://127.0.0.1:6333"


@dataclass(frozen=True, slots=True)
class RagExampleSpec:
    example_id: str
    example_dir: Path
    rag_command: list[str]
    rag_env: dict[str, str]
    config_name: str
    poison_mode: str
    native_backend: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vx_executable() -> Path:
    candidate = _repo_root() / ".venv" / "bin" / "vx"
    if candidate.is_file():
        return candidate
    vx_on_path = shutil.which("vx")
    if vx_on_path:
        return Path(vx_on_path)
    pytest.skip("vx CLI not found (.venv/bin/vx or PATH)")


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _uses_qdrant_server(rag_example_spec: RagExampleSpec) -> bool:
    return (
        rag_example_spec.example_id == "medium_qdrant"
        and rag_example_spec.poison_mode == "native"
    )


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return probe.returncode == 0


def _wait_for_http(url: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "unknown"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    pytest.skip(f"RAG target not ready at {url} within {timeout_s}s ({last_error})")


def _ollama_has_model(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    models = payload.get("models")
    if not isinstance(models, list):
        return False
    names = {
        str(item.get("name", "")).strip() for item in models if isinstance(item, dict)
    }
    return model in names


@pytest.fixture(scope="session")
def qdrant_server_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip(
            "Docker unavailable (required for medium_qdrant:native e2e); "
            "install Docker and pull qdrant/qdrant"
        )

    port = QDRANT_HOST_PORT
    base_url = f"http://127.0.0.1:{port}"
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:6333",
            QDRANT_DOCKER_IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if run.returncode != 0:
        detail = (run.stderr or run.stdout or "").strip()
        pytest.skip(f"Failed to start Qdrant container: {detail}")

    container_id = (run.stdout or "").strip()
    if not container_id:
        pytest.skip("docker run did not return a container id")

    try:
        _wait_for_http(f"{base_url}/", timeout_s=QDRANT_SERVER_READY_TIMEOUT_S)
        yield base_url
    finally:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            timeout=30,
            check=False,
        )


@pytest.fixture(scope="session")
def ollama_available() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5):
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"Ollama unavailable at {OLLAMA_BASE_URL}: {exc}")
    if not _ollama_has_model(OLLAMA_MODEL):
        pytest.skip(
            f"Ollama model {OLLAMA_MODEL!r} not installed "
            f"(run: ollama pull {OLLAMA_MODEL})"
        )


def _require_native_deps(backend: str) -> None:
    pytest.importorskip("sentence_transformers")
    if backend == "chroma":
        pytest.importorskip("chromadb")
    elif backend == "faiss":
        pytest.importorskip("faiss")
    elif backend == "qdrant":
        pytest.importorskip("qdrant_client")


def _example_dir(relative: str) -> Path:
    return _repo_root() / "rag_examples" / relative


def _rag_python(example_dir: Path) -> str:
    local_venv = example_dir / ".venv" / "bin" / "python"
    if local_venv.is_file():
        return str(local_venv)
    return sys.executable


def _build_rag_command(example_dir: Path, script: str) -> list[str]:
    return [_rag_python(example_dir), script]


RAG_EXAMPLES: tuple[RagExampleSpec, ...] = (
    RagExampleSpec(
        example_id="small_in_memory",
        example_dir=_example_dir("small/rag_01_in_memory_en"),
        rag_command=_build_rag_command(
            _example_dir("small/rag_01_in_memory_en"),
            "small_rag.py",
        ),
        rag_env={},
        config_name="ollama-smoke.yaml",
        poison_mode="file_text",
    ),
    RagExampleSpec(
        example_id="medium_chroma",
        example_dir=_example_dir("medium/rag_01_nq_chroma_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_01_nq_chroma_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke.yaml",
        poison_mode="file_text",
    ),
    RagExampleSpec(
        example_id="medium_chroma",
        example_dir=_example_dir("medium/rag_01_nq_chroma_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_01_nq_chroma_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke-native-poisoner.yaml",
        poison_mode="native",
        native_backend="chroma",
    ),
    RagExampleSpec(
        example_id="medium_faiss",
        example_dir=_example_dir("medium/rag_02_nq_faiss_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_02_nq_faiss_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke.yaml",
        poison_mode="file_text",
    ),
    RagExampleSpec(
        example_id="medium_faiss",
        example_dir=_example_dir("medium/rag_02_nq_faiss_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_02_nq_faiss_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke-native-poisoner.yaml",
        poison_mode="native",
        native_backend="faiss",
    ),
    RagExampleSpec(
        example_id="medium_qdrant",
        example_dir=_example_dir("medium/rag_03_nq_qdrant_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_03_nq_qdrant_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke.yaml",
        poison_mode="file_text",
    ),
    RagExampleSpec(
        example_id="medium_qdrant",
        example_dir=_example_dir("medium/rag_03_nq_qdrant_en"),
        rag_command=_build_rag_command(
            _example_dir("medium/rag_03_nq_qdrant_en"),
            "nq_rag.py",
        ),
        rag_env={"RAG_CONFIG": "config.smoke.json"},
        config_name="ollama-smoke-native-poisoner-server.yaml",
        poison_mode="native",
        native_backend="qdrant",
    ),
)


@pytest.fixture
def rag_example_spec(request: pytest.FixtureRequest) -> RagExampleSpec:
    return request.param


@pytest.fixture
def rag_target_process(
    rag_example_spec: RagExampleSpec,
    ollama_available: None,
    request: pytest.FixtureRequest,
) -> Iterator[subprocess.Popen[bytes]]:
    if rag_example_spec.poison_mode == "native":
        assert rag_example_spec.native_backend is not None
        _require_native_deps(rag_example_spec.native_backend)

    if _port_is_open(TARGET_HOST, TARGET_PORT):
        pytest.skip(
            f"Port {TARGET_PORT} is already in use; "
            "e2e tests require an exclusive RAG target on :8080"
        )

    example_dir = rag_example_spec.example_dir
    if not example_dir.is_dir():
        pytest.skip(f"RAG example directory missing: {example_dir}")

    config_path = example_dir / "scan_configs_examples" / rag_example_spec.config_name
    if not config_path.is_file():
        pytest.skip(f"Scan config missing: {config_path}")

    if importlib.util.find_spec("uvicorn") is None:
        pytest.skip("uvicorn not installed for RAG example subprocess")

    env = os.environ.copy()
    env.update(rag_example_spec.rag_env)
    if _uses_qdrant_server(rag_example_spec):
        qdrant_url = request.getfixturevalue("qdrant_server_url")
        env["QDRANT_URL"] = qdrant_url

    proc = subprocess.Popen(
        rag_example_spec.rag_command,
        cwd=example_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_http(TARGET_HEALTH_URL, timeout_s=READINESS_TIMEOUT_S)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def render_qdrant_server_scan_config(
    example_dir: Path,
    config_name: str,
    qdrant_url: str,
) -> Path:
    template = example_dir / "scan_configs_examples" / config_name
    content = template.read_text(encoding="utf-8")
    rendered = content.replace(QDRANT_SCAN_CONFIG_PLACEHOLDER_URL, qdrant_url)
    if rendered == content:
        raise ValueError(
            f"scan config {template} missing placeholder "
            f"{QDRANT_SCAN_CONFIG_PLACEHOLDER_URL!r}"
        )
    out = example_dir / "scan_configs_examples" / ".e2e-qdrant-server.yaml"
    out.write_text(rendered, encoding="utf-8")
    return out


def run_vx_scan(
    example_dir: Path,
    config_name: str,
    *,
    config_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if config_path is not None:
        config_arg = str(config_path.resolve())
    else:
        config_arg = str(Path("scan_configs_examples") / config_name)
    return subprocess.run(
        [
            str(_vx_executable()),
            "scan",
            "--config",
            config_arg,
        ],
        cwd=example_dir,
        capture_output=True,
        text=True,
        timeout=SCAN_TIMEOUT_S,
        check=False,
    )
