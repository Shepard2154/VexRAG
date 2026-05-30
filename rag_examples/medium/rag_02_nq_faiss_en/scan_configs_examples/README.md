# Scan configs (NQ + FAISS)

Start the target first: `python3 nq_rag.py` (from the parent directory). First run builds the FAISS index from NQ (see `config.json` `nq_dataset.limit`).

## Recommended (Ollama)

| Config | Models | Cases |
|--------|--------|-------|
| **`ollama-default.yaml`** | `llama3:8b`, `nomic-embed-text:latest` | Full NQ case set |
| **`ollama-smoke.yaml`** | `llama3:8b` | 2 hijack + 1 poisoned |

```bash
vx scan --config scan_configs_examples/ollama-default.yaml
vx scan --config scan_configs_examples/ollama-smoke.yaml
```

Poison files are written under `../poisoned_contexts/` and indexed on the next target request.

**Faster target:** `RAG_CONFIG=config.smoke.json python3 nq_rag.py` (100 NQ passages instead of 2000).

## Advanced

[`advanced/chain-semantic-only.yaml`](advanced/chain-semantic-only.yaml) — composite eval without LLM judge (embeddings only).
