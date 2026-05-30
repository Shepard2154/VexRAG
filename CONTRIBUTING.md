# Contributing

## Development Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

See the [Development setup](README.md#development-setup) section in the root README for a quick overview.

## Tests

Layout mirrors the `vexrag/` package under `tests/` (e.g. `tests/core/`, `tests/cli/`, `tests/e2e/`). Shared fixtures live in `tests/conftest.py`; CLI/scan stubs in `tests/mocks.py`.

```bash
poe test              # fast unit tests (excludes integration and e2e)
poe test-integration  # wiring tests (mocked HTTP, etc.)
poe test-e2e          # full vx scan against rag_examples (see below)
poe cov               # unit tests with coverage (excludes e2e)
```

### E2E smoke scans (`poe test-e2e`)

Prerequisites:

- Ollama running at `http://localhost:11434` with `llama3:8b`:

  ```bash
  ollama pull llama3:8b
  ```

- Repo dev install: `pip install -e ".[dev]"` and `.venv/bin/vx` available.
- Port `8080` free (all e2e cases share one RAG target).
- RAG example dependencies installed in the example directory or current Python env (see each `rag_examples/*/requirements.txt`).
- For **native** vector-DB poisoner cases (`ollama-smoke-native-poisoner.yaml`), install optional extras as needed:

  ```bash
  pip install -e ".[dev,sentence-transformers,faiss,chroma,qdrant]"
  ```

Expect ~7 sequential smoke scans (several minutes each with LLM calls). Tests skip cleanly when Ollama, models, deps, or port `8080` are unavailable — normal in CI without GPU/Ollama.

The **`medium_qdrant:native`** case uses a Qdrant **server** (`qdrant/qdrant` via Docker) so the RAG target and `vx scan` do not contend for an embedded `qdrant_data/` lock. Prerequisites:

- Docker installed and running; image `qdrant/qdrant` (pulled on first run).
- Same Ollama, `vx`, port `8080`, and native extras as other native cases.

```bash
poe test-e2e -k "medium_qdrant:native"
```

Without Docker, that single test is **skipped**; the other six e2e cases are unchanged.

## Code Quality

Run checks before committing:

```bash
ruff check .
ruff format --check .
```

Auto-fix issues:

```bash
ruff check --fix .
ruff format .
```

## Pre-commit Hooks

Install hooks once per clone:

```bash
pre-commit install
```

Run hooks manually on all files:

```bash
pre-commit run --all-files
```

## Git Commits

- **Commits are human-only.** The Cursor agent must not run `git commit` or `git push` unless you explicitly ask it to.
- When changes are ready, review the diff and commit locally yourself.

## Commit Scope

- Keep changes minimal and focused on one concern.
- Avoid unrelated refactors in the same commit.
