<p align="center">
  <img src="https://raw.githubusercontent.com/Shepard2154/VexRAG/v0.2.1/assets/logo.png" alt="VexRAG" width="300">
</p>

<p align="center">
  <a href="https://github.com/Shepard2154/VexRAG"><img src="https://img.shields.io/badge/project-in%20development-F59E0B?style=flat-square" alt="Project: in development" height="28"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python 3.11+" height="28"></a>
  <a href="https://pepy.tech/projects/vexrag"><img src="https://static.pepy.tech/personalized-badge/vexrag?period=total&units=ABBREVIATION&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" height="28"></a>
  <a href="https://t.me/vexrag"><img src="https://img.shields.io/badge/chat-join-blue?style=flat-square&logo=telegram" alt="Telegram chat" height="28"></a>
</p>

Most RAG security tools focus on jailbreaking or prompt injection. VexRAG is different: it injects poisoned passages directly into the retrieval index and measures the system’s answer functional correctness under adversarial data manipulation.

## Threat model — when to use VexRAG

VexRAG is for **security testing your own RAG** in a **controlled, isolated environment** (not production).

**Use VexRAG if:**

- Retrieval data may be **untrusted or partially adversarial** (uploads, crawls, third-party corpora).
- You need to measure behavior when an attacker can **poison or skew the index**.
- You are building **trust-aware RAG** and want evidence of resilience, not only prompt-level guards.

**You probably do not need VexRAG if:**

- Every indexed document is fully trusted, ingestion is strictly controlled, and you accept that risk without red-team validation.
- You only care about **query-time** prompt injection or jailbreaks (VexRAG targets **retrieval and corpus** attacks).

## Science-first approach

VexRAG implements **paper-backed** attacks, not ad-hoc heuristics:

<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="left">Paper</th>
      <th align="left">Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>PoisonedRAG</strong></td>
      <td><a href="https://arxiv.org/abs/2402.07867">arXiv:2402.07867</a></td>
      <td>Poisoning the retrieval corpus</td>
    </tr>
    <tr>
      <td><strong>HijackRAG</strong></td>
      <td><a href="https://arxiv.org/abs/2410.22832">arXiv:2410.22832</a></td>
      <td>Hijacking retrieved contexts</td>
    </tr>
  </tbody>
</table>

See [`vexrag/attack_algorithms/`](vexrag/attack_algorithms/) for implementation details and fidelity notes.

> **Warning — real corpus mutation**  
> Scans write poisoned passages into the target retrieval index. Configs default to `cleanup: true`, but interrupted runs may still leave poison behind.  
> Never target production or shared indexes. Back up the retrieval database before testing on your own data.

## Versioning Policy

VexRAG is an early-stage library. Until `1.0.0`, treat **any release as potentially breaking** — configs, CLI flags, and APIs may change without a major version bump.

When we deprecate public functionality, it stays available for **two minor releases** before removal (e.g. deprecated in `0.3.0`, removed in `0.5.0`).

From `1.0.0` onward we plan to follow [SemVer](https://semver.org/) (breaking changes in major releases only).

## Prerequisites

Use a sample RAG target from [rag_examples](rag_examples/README.md): start the example app (each folder’s README has the command), then scan it with `vx`.

For the default Ollama-based configs you need **Python 3.11+** and a running Ollama daemon. CI and releases are tested on **3.11** only; newer versions may work but are not officially supported yet.

```bash
python --version  # 3.11+ required; 3.11 tested in CI
ollama list
```

Install/pull Ollama models for scan configs:

```bash
ollama pull llama3:8b
```

For full benchmarks (`ollama-default.yaml` and some advanced configs), also pull `nomic-embed-text:latest`.

All [rag_examples](rag_examples/README.md) targets default to `http://localhost:8080`; run one example at a time.

## Installation

```bash
pip install vexrag
```

Optional extras (install what your stack needs):

```bash
pip install "vexrag[qdrant]"
pip install "vexrag[chroma]"
pip install "vexrag[faiss]"
pip install "vexrag[sentence-transformers]"
```

The `sentence-transformers` extra enables the in-process embedding provider in scan configs; model weights download from Hugging Face on first use.

Verify the CLI:

```bash
vx --help
```

## Run a scan

```bash
vx scan --config path/to/scan.yaml
```

Use sample configs from `rag_examples/` as a starting point.

## First successful scan

From `rag_examples/small/rag_01_in_memory_en`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install vexrag
pip install -r requirements.txt
python3 small_rag.py
vx scan --config scan_configs_examples/ollama-smoke.yaml
```

Expected outcome:
- `python3 small_rag.py` serves the target API on `http://localhost:8080` (embeddings via Hugging Face; first run may download model weights).
- `vx scan` finishes with a scan report. Smoke config needs only `ollama pull llama3:8b`.

## Project roadmap

### Done
- [x] Implementation of PoisonedRAG (**arXiv:** [2402.07867](https://arxiv.org/abs/2402.07867))
- [x] Implementation of HijackRAG (**arXiv:** [2410.22832](https://arxiv.org/abs/2410.22832))
- [x] Automatic generation of attack cases for both methods
- [x] Support for vLLM and Ollama
- [x] Simple RAG examples for quick onboarding to VexRAG
- [x] Support for Qdrant, FAISS, Chroma, and file-based retrieval backends
- [x] Codebase hardening: refactors, typing, tooling, *removing AI slop*

### In Progress
- [ ] Stable Python API to run scans and generate cases from code, not only via `vx`

### Ideas / Backlog
- [ ] Expand red-team methods in VexRAG
- [ ] Expand supported retrieval backends
- [ ] Implement a web version of VexRAG

## Feedback

Feel free to [open a GitHub issue](https://github.com/Shepard2154/VexRAG/issues) for bugs, questions, or attack methods you would like to see in VexRAG. Pull requests and local development notes are in [CONTRIBUTING.md](CONTRIBUTING.md).
