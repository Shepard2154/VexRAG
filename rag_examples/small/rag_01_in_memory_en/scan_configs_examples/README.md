# Scan configs (small in-memory RAG)

Start the target first: `python3 small_rag.py` (from the parent directory).

## Recommended (Ollama only)

| Config | Models | Cases |
|--------|--------|-------|
| **`ollama-default.yaml`** | `llama3:8b`, `nomic-embed-text:latest` | Full set (9 hijack + 2 poisoned) |
| **`ollama-smoke.yaml`** | `llama3:8b` only | Smoke (2 hijack + 1 poisoned), faster |

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text:latest   # required for ollama-default.yaml only

vx scan --config scan_configs_examples/ollama-default.yaml
vx scan --config scan_configs_examples/ollama-smoke.yaml
```

## Case files

- `cases/hijackrag-cases.yaml`, `cases/poisonedrag-cases.yaml` — full benchmark
- `cases/smoke-hijackrag.yaml`, `cases/smoke-poisonedrag.yaml` — quick runs

## Advanced (optional)

Extra providers, evaluators, or heavier models — see [`advanced/README.md`](advanced/README.md).
