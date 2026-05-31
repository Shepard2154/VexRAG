# Contributing

## Development setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Tests

| Command | Scope |
| --- | --- |
| `poe test` | Unit tests (default for PRs) |
| `poe test-integration` | Wiring tests with mocked HTTP |
| `poe test-e2e` | Full `vx scan` smoke tests |
| `poe cov` | Unit tests with coverage |

E2E tests require Ollama with `llama3:8b`, port `8080`, and deps from `rag_examples/` (see [README](README.md)). They skip cleanly when prerequisites are missing.

## Code quality

CI runs the same checks as a local PR gate:

```bash
poe check   # ruff + mypy
poe test
```

Auto-fix formatting and lint issues:

```bash
poe fix
```

Or run hooks manually: `pre-commit run --all-files`.

## Pull requests

- Keep changes minimal and focused on one concern.
- Avoid unrelated refactors in the same PR.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml`.
2. Merge to `master` and wait for CI to pass.
3. `git tag vX.Y.Z && git push origin vX.Y.Z`.

The [package-release workflow](.github/workflows/package-release.yml) builds `dist/*`, creates a GitHub Release, and publishes stable tags to PyPI. Prerelease tags (e.g. `v1.0.0-rc1`) skip PyPI.
