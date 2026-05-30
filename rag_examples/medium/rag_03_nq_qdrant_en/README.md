# RAG 03: NQ-style corpus + Qdrant

## Corpus

On startup, the service initializes a Qdrant collection from the **Natural Questions** dataset (`nq_dataset` in `config.json`).
Base retrieval contexts are loaded from NQ and indexed directly in Qdrant, not from local `.txt` files.

## Quick start

From `rag_examples/medium/rag_03_nq_qdrant_en`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 nq_rag.py
```

Embeddings and vectors are stored under `qdrant_data/` (created on first run). Override with `QDRANT_DIR`.
Use embedded storage (`QdrantClient(path=...)`) when a single process owns the index. Set `QDRANT_URL` (e.g. `http://127.0.0.1:6333`) when VexRAG native corpus poisoning runs against the same Qdrant server in parallel with this target.
Poison payload files are written into `poisoned_contexts/` and then indexed into the same Qdrant collection.

Default service URL: `http://localhost:8080`  
Endpoint: `POST /model/context-based-response` — JSON body `{"query": "..."}`; response includes `answer` and `contexts`.

## Smoke vs full mode

| Mode | Config | Corpus source | Typical use |
|------|--------|---------------|-------------|
| **Full** | `config.json` (default) | Hugging Face NQ (`google-research-datasets/natural_questions`, 2000 passages) | Realistic index size, first run downloads NQ |
| **Smoke** | `RAG_CONFIG=config.smoke.json` | Local `benchmark.jsonl` (10 passages with embedded answers) | Fast startup for scans and attack demos |

Smoke mode sets `nq_dataset.enabled: false` so the service skips HF download and uses `benchmark.jsonl` instead.
Both modes reuse the Qdrant collection when the corpus fingerprint and embedding model match stored metadata.

```bash
RAG_CONFIG=config.smoke.json python3 nq_rag.py
```

## Docker

```bash
docker build -t rag-03-nq-qdrant-en .
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/poisoned_contexts:/app/poisoned_contexts" \
  -v "$PWD/qdrant_data:/app/qdrant_data" \
  rag-03-nq-qdrant-en
```

Use `--add-host=host.docker.internal:host-gateway` on Linux so the container can reach Ollama on the host.

## LLM (Ollama)

Same overrides as the small example:

```bash
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=llama3:8b \
python3 nq_rag.py
```

## VexRAG scan

With the service running:

```bash
vx scan --config scan_configs_examples/ollama-smoke.yaml
```

See `scan_configs_examples/README.md`.

Bundled cases use the same **questions and gold answers** as the local `benchmark.jsonl` / NQ slice, so retrieved passages can support the scan queries. For `PoisonedRAG`, the chain config runs **LLM judge** on step 2 only, because embedding-only similarity often labels a poisoned factual answer as “not an attack” when it is equally close to both the clean and poison claims in vector space.
