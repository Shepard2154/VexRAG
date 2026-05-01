# RAG 01: In-Memory Baseline

RAG service compatible with VexRAG.

**In-memory** here means the embedding matrix and retrieval run entirely in RAM (no external vector database). The service starts from passages in `benchmark.jsonl` and `contexts/`, then refreshes `contexts/poisonedrag_*.txt` before answering so new poisoned passages can enter the in-process NumPy index.

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

## LLM (Ollama and vLLM)

By default, the app and baseline scan use Ollama at `http://localhost:11434` with `llama3:8b`. Ollama must be running and the model pulled locally.

You can override the app config with environment variables:

```bash
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=llama3:8b \
python3 small_rag.py
```

## VexRAG Scan

Start the demo service first, then run the VexRAG scan from the repository root:

```bash
vx scan --config "RAG examples/small/rag_01_in_memory_en/vexrag-poisonedrag-llm-judge-ollama-llama3-8b.yaml"
```

The included scan configs target `http://localhost:8080/model/context-based-response` and run the inline case plus cases from `poisonedrag-cases.yaml`.
- `vexrag-poisonedrag-llm-judge-ollama-llama3-8b.yaml` uses `evaluation.strategy: llm_judge`.
- `vexrag-poisonedrag-semantic-similarity-ollama-llama3-8b.yaml` uses `evaluation.strategy: semantic_similarity` and enables deterministic-style scoring.
- `vexrag-poisonedrag-llm-judge-vllm-qwen3-30b-a3b-instruct-2507.yaml` uses vLLM (`Qwen/Qwen3-30B-A3B-Instruct-2507`) via OpenAI-compatible API.
With `scan.corpus_poisoning.path: ./contexts`, VexRAG writes generated poisoned texts as `poisonedrag_*.txt` files into `contexts/` for `file_text` corpus poisoning. Set `scan.corpus_poisoning.cleanup: true` to remove these poisoned files after each scan case. With the default configs, make sure `llama3:8b` is available locally or change the model fields in both config files.

## Docker Run

Inside a container, `localhost` is not the host. If Ollama runs on the machine (not in the same container), point the app at the host:

```bash
docker build -t rag-01-in-memory-en .
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/contexts:/app/contexts" \
  rag-01-in-memory-en
```

On Linux, `host.docker.internal` requires `--add-host=host.docker.internal:host-gateway` (shown above). Override `OLLAMA_BASE_URL` if Ollama listens elsewhere.
After the container starts, run the VexRAG scan from the repository root with `vexrag-poisonedrag-llm-judge-ollama-llama3-8b.yaml` (or `vexrag-poisonedrag-semantic-similarity-ollama-llama3-8b.yaml`). The scan runs on the host, so it keeps using `http://localhost:11434` for Ollama. Keep the `contexts/` bind mount when scanning a containerized app; otherwise VexRAG writes poisoned files to the host folder while the container reads its baked-in corpus.
