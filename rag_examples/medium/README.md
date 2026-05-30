# Medium RAG Examples

The `medium` folder contains RAG demos with external vector storage and larger retrieval corpora than `small`.

- `rag_01_nq_chroma_en` — Natural Questions-style QA with passages stored in **ChromaDB** (persistent).
- `rag_02_nq_faiss_en` — Natural Questions-style QA with passages stored in **FAISS**.
- `rag_03_nq_qdrant_en` — Natural Questions-style QA with passages stored in **Qdrant**.

Each example provides two smoke scan configs under `scan_configs_examples/`:

- **`ollama-smoke.yaml`** — `file_text` poisoning (writes to `poisoned_contexts/`; target reloads on request).
- **`ollama-smoke-native-poisoner.yaml`** — native vector-DB poisoning via VexRAG (`backend: chroma|faiss|qdrant`), using the same persist paths as the target.
