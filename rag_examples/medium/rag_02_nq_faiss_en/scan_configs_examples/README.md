# Scan configs (NQ + FAISS)

Start the target first: `python3 nq_rag.py` (from the parent directory). First run builds the FAISS index from NQ (see `config.json` `nq_dataset.limit`).

## Recommended (Ollama)

| Config | Models | Corpus poisoning | Cases |
|--------|--------|------------------|-------|
| **`ollama-default.yaml`** | `llama3:8b`, `nomic-embed-text:latest` | `file_text` → `../poisoned_contexts/` | Full NQ case set |
| **`ollama-smoke.yaml`** | `llama3:8b` | `file_text` → `../poisoned_contexts/` | 2 hijack + 1 poisoned |
| **`ollama-smoke-native-poisoner.yaml`** | `llama3:8b`, `all-MiniLM-L6-v2` (embed) | **Native FAISS** → `../faiss_data/` | 2 hijack + 1 poisoned |

```bash
vx scan --config scan_configs_examples/ollama-default.yaml
vx scan --config scan_configs_examples/ollama-smoke.yaml
vx scan --config scan_configs_examples/ollama-smoke-native-poisoner.yaml
```

- **`ollama-smoke.yaml`** — VexRAG writes poison text files under `../poisoned_contexts/`; the target reloads and indexes them on the next request.
- **`ollama-smoke-native-poisoner.yaml`** — VexRAG appends adversarial vectors directly to the FAISS index (`scan.corpus_poisoning.backend: faiss`). No poison files are required; the target reloads native poison from disk on retrieve.

**Faster target:** `RAG_CONFIG=config.smoke.json python3 nq_rag.py` (local `benchmark.jsonl`, no HF download).

## Advanced

[`advanced/chain-semantic-only.yaml`](advanced/chain-semantic-only.yaml) — composite eval without LLM judge (embeddings only).
