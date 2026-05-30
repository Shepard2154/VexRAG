import pytest

from tests.e2e.conftest import (
    RAG_EXAMPLES,
    _uses_qdrant_server,
    render_qdrant_server_scan_config,
    run_vx_scan,
)


def _format_scan_failure(result) -> str:
    stderr = result.stderr or ""
    for line in stderr.splitlines():
        if "Error:" in line:
            return line.strip()
    lines = [line for line in stderr.splitlines() if line.strip()]
    if lines:
        tail = "\n".join(lines[-15:])
        return f"exit {result.returncode}\n{tail}"
    stdout_lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if stdout_lines:
        return f"exit {result.returncode}\nstdout tail:\n" + "\n".join(
            stdout_lines[-15:]
        )
    return f"exit {result.returncode} (no stderr/stdout)"


@pytest.mark.e2e
@pytest.mark.parametrize(
    "rag_example_spec",
    RAG_EXAMPLES,
    ids=[f"{spec.example_id}:{spec.poison_mode}" for spec in RAG_EXAMPLES],
    indirect=True,
)
class TestRagExampleSmokeScan:
    def test_vx_scan_completes_with_report_markers(
        self,
        rag_example_spec,
        rag_target_process,
        request: pytest.FixtureRequest,
    ) -> None:
        del rag_target_process

        if _uses_qdrant_server(rag_example_spec):
            qdrant_url = request.getfixturevalue("qdrant_server_url")
            config_path = render_qdrant_server_scan_config(
                rag_example_spec.example_dir,
                rag_example_spec.config_name,
                qdrant_url,
            )
            result = run_vx_scan(
                rag_example_spec.example_dir,
                rag_example_spec.config_name,
                config_path=config_path,
            )
        else:
            result = run_vx_scan(
                rag_example_spec.example_dir,
                rag_example_spec.config_name,
            )

        assert result.returncode == 0, _format_scan_failure(result)
        assert "Final verdict:" in result.stdout
        assert "VexRAG Scan" in result.stdout
