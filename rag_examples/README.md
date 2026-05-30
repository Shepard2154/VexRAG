# RAG examples

Self-contained sample RAG targets for onboarding and evaluation with VexRAG.

| Tier | Examples | Scan configs |
|------|----------|--------------|
| **small** | `small/rag_01_in_memory_en` | `ollama-default.yaml`, `ollama-smoke.yaml` |
| **medium** | NQ + Chroma / FAISS / Qdrant | Same layout per example |
| **huge** | StackOverflow ingest scripts only (no `vx scan` target yet) |

**Quick start:** run the target (`python3 small_rag.py` or `python3 nq_rag.py`), then:

```bash
vx scan --config scan_configs_examples/ollama-smoke.yaml
```

Each example’s `scan_configs_examples/README.md` lists full vs smoke scans and optional `advanced/` configs.
