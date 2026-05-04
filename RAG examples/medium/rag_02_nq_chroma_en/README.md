# RAG 02: NQ-style corpus + ChromaDB

## Corpus

Questions and short answers follow the **Natural Questions** style (open-domain question + concise answer). Long retrieval passages live under `contexts/` as `{id}.txt`, matching `benchmark.jsonl`.

The bundled `benchmark.jsonl` is a **small local subset** for demos. For full-scale NQ, export passages from the [Natural Questions dataset](https://github.com/google-research-datasets/natural-questions) and replace these files.

## Quick start

From `RAG examples/medium/rag_02_nq_chroma_en`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 nq_rag.py
```

Embeddings and vectors are stored under `chroma_data/` (created on first run). Override with `CHROMA_DIR`.

Default service URL: `http://localhost:8080`  
Endpoint: `POST /model/context-based-response` — JSON body `{"query": "..."}`; response includes `answer` and `contexts`.

## Docker

```bash
docker build -t rag-02-nq-chroma-en .
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/contexts:/app/contexts" \
  -v "$PWD/chroma_data:/app/chroma_data" \
  rag-02-nq-chroma-en
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

With the service running, from `scan_configs_examples/`:

```bash
vx scan --config <path_to_yaml>
```
