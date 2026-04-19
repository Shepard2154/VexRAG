# RAG 01: In-Memory Baseline

RAG service compatible with VexRAG.

**In-memory** here means the embedding matrix and retrieval run entirely in RAM (no external vector database). Passages are read from `benchmark.jsonl` and `contexts/` once at startup, then search uses an in-process NumPy index.

## Corpus source

Retrieval passages are adapted from the **`db_records`** list used in *RAG-Driven Generative AI* (2nd ed., Chapter 1). Each list entry is one file under `contexts/`.

**Source:** [RAG-Driven Generative AI (2nd ed.)](https://github.com/Denis2054/RAG-Driven-Generative-AI-2nd-Edition).

## Quick Start

From `RAG examples/small/rag_01_in_memory_en`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 small_rag.py
```

Default service URL: `http://localhost:9003`  
Answer endpoint: `POST /model/with-context` — response includes `answer` and `contexts` (retrieved passages).

## LLM (Ollama)

Configure `llm.ollama` in `config.json` (`base_url`, `model`, `temperature`). Ollama must be running and the model pulled locally.

## Docker Run

Inside a container, `localhost` is not the host. If Ollama runs on the machine (not in the same container), point the app at the host:

```bash
docker build -t rag-01-in-memory-en .
docker run --rm -p 9003:9003 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-01-in-memory-en
```

On Linux, `host.docker.internal` requires `--add-host=host.docker.internal:host-gateway` (shown above). Override `OLLAMA_BASE_URL` if Ollama listens elsewhere.
