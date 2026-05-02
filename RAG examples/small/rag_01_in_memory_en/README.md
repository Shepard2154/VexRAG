# RAG 01: In-Memory Baseline

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

Default service URL: `http://localhost:8080`  
Answer endpoint: `POST /model/context-based-response` — response includes `answer` and `contexts` (retrieved passages).

## Docker Run

From this directory, with Ollama on the host (not in the container):

```bash
docker build -t rag-01-in-memory-en .
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/contexts:/app/contexts" \
  rag-01-in-memory-en
```

Use `--add-host=...` on Linux. Change `OLLAMA_BASE_URL` if Ollama listens elsewhere.

## LLM (Ollama and vLLM)

By default, the app and baseline scan use Ollama at `http://localhost:11434` with `llama3:8b`. Ollama must be running and the model pulled locally.

You can override the app config with environment variables:

```bash
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=llama3:8b \
python3 small_rag.py
```

## VexRAG Scan

With the demo service running, pick a YAML under `scan_configs_examples/` and run:

```bash
vx scan --config <path_to_yml>
```
