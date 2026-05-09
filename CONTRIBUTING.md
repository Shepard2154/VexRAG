# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

## Commit Scope

- Keep changes minimal and focused on one concern.
- Avoid unrelated refactors in the same commit.
