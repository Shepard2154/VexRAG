# VexRAG

![Project: in development](https://img.shields.io/badge/project-in%20development-F59E0B?style=for-the-badge)

A toolkit for assessing the **functional correctness** of retrieval-augmented generation (RAG) systems under attack conditions.

**Sample RAG stacks** for getting started: [RAG examples](RAG%20examples/README.md).

## Quickstart

### Prerequisites

```bash
python --version  # requires 3.11+
ollama list
```

Install/pull required Ollama models:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text:latest
```

You also need a running target API endpoint (for the small example: `http://localhost:8080`).

### 1) Install VexRAG

```bash
pip install vexrag
```

For vector DB-specific extras:

```bash
pip install "vexrag[qdrant]"
pip install "vexrag[chroma]"
pip install "vexrag[faiss]"
```

### 2) Verify installation

```bash
vx --help
```

### 3) Run a scan from config

```bash
vx scan --config path/to/scan.yml
```

Use sample configs from `RAG examples/` as a starting point.

### 4) First successful scan (small local example)

From `RAG examples/small/rag_01_in_memory_en`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python small_rag.py
vx scan --config scan_configs_examples/vexrag-chain-hijack-then-poisoned-semantic-ollama-nomic.yaml
```

Expected outcome:
- `small_rag.py` serves the target API on `http://localhost:8080`.
- `vx scan` completes and prints a scan report with attack/evaluation results (no connection/preflight errors).

## Project roadmap

Canonical checklist: [notes/TODO.md](notes/TODO.md).

## Done
- [x] Small RAG (in-memory)
- [x] PoisonedRAG target scan pipeline with core target, scan, and evaluation contracts
- [x] PoisonedRAG CLI scan flow wired from YAML config with multi-context poisoning runs
- [x] Core package facade exports clarified for shared APIs
- [x] StackOverflow XML/TSV to Qdrant ingestion scripts for large dataset indexing
- [x] PoisonedRAG generation improvements: poisoning styles, corpusN payloads, and query-prefixed adversarial outputs
- [x] Automatic attack case generation and consolidated example scan configs
- [x] HijackRAG attack support with CLI `generate-cases`
- [x] vLLM target/provider support for scan execution
- [x] Core modularization for config/retrieval/runtime

## In Progress
- [ ] PoisonedRAG hardening: broaden scenario coverage, stabilize metrics, and add end-to-end validation runs
- [ ] Medium RAG examples stabilization across vector DB backends and multi-attack eval flow

## Next
- [ ] Finalize full end-to-end runnable demo for the huge StackOverflow + Qdrant pipeline
- [ ] Promote selected `wip` milestones to stable feature/documented workflow status

## Ideas / Backlog
- [ ] Red-team testing methods for API-interacting RAG services (local RAG targets)
- [ ] Red-team testing methods for the VexRAG CLI (local RAG targets)
