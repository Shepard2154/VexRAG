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
vx scan --config "RAG examples/small/rag_01_in_memory_en/scan_configs_examples/vexrag-poisonedrag-llm-judge-ollama-llama3-8b.yaml"
```

The included scan configs are kept in `scan_configs_examples/` and target `http://localhost:8080/model/context-based-response`.
- `scan_configs_examples/vexrag-poisonedrag-llm-judge-ollama-llama3-8b.yaml` uses `evaluation.strategy: llm_judge`.
- `scan_configs_examples/vexrag-poisonedrag-llm-judge-vllm-gemma-3-27b-it-original.yaml` is the Gemma baseline (`poisoning_style: original`).
- `scan_configs_examples/vexrag-poisonedrag-llm-judge-vllm-gemma-3-27b-it-aggressive.yaml` keeps the same setup with stronger injection (`poisoning_style: aggressive`).
- `scan_configs_examples/vexrag-poisonedrag-semantic-similarity-vllm-gemma-3-27b-it-original.yaml` uses `evaluation.strategy: semantic_similarity`.
- `scan_configs_examples/vexrag-poisonedrag-llm-judge-vllm-qwen3-30b-a3b-instruct-2507.yaml` uses vLLM (`Qwen/Qwen3-30B-A3B-Instruct-2507`) via OpenAI-compatible API.
Case files are grouped in `scan_configs_examples/cases/`.
With `scan.corpus_poisoning.path: ../contexts`, VexRAG writes generated poisoned texts as `poisonedrag_*.txt` files into `contexts/` for `file_text` corpus poisoning. Set `scan.corpus_poisoning.cleanup: true` to remove these poisoned files after each scan case.

### Automatic PoisonedRAG Cases

You can auto-generate a `cases:` file compatible with `attack.poisonedrag.case_files`:

```bash
vx generate-cases \
  --config "RAG examples/small/rag_01_in_memory_en/scan_configs_examples/vexrag-poisonedrag-llm-judge-vllm-gemma-3-27b-it-original.yaml" \
  --output "RAG examples/small/rag_01_in_memory_en/scan_configs_examples/cases/poisonedrag-cases-auto.yaml" \
  --count 6 \
  --topic "retrieval governance and data freshness" \
  --overwrite
```

Then reference it in your scan config:

```yaml
attack:
  poisonedrag:
    poisoning_style: original
    case_files:
      - ./cases/poisonedrag-cases-auto.yaml
```

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
After the container starts, run the VexRAG scan from the repository root with one of the files from `RAG examples/small/rag_01_in_memory_en/scan_configs_examples/`. The scan runs on the host, so it keeps using host model endpoints (for example `http://localhost:11434` for Ollama). Keep the `contexts/` bind mount when scanning a containerized app; otherwise VexRAG writes poisoned files to the host folder while the container reads its baked-in corpus.
