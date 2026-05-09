# VexRAG CLI

Command-line tool for assessing the security of your local RAG system.

## Scan Config

Run scans with:

```shell
vx scan --config config.yaml
```

Each scan config lists one or more attacks under `attacks` (in order). A single
entry is a normal scan; multiple entries run as one chain (same aggregate
report, per-step summary in the CLI).

Shared sections (`target_system`, `evaluation`, `scan`) apply to every step.
Optional per-step `scan` / `evaluation` / `evaluations` mappings are deep-merged
over the root blocks for that step only.

For several evaluation backends on the same answers, use top-level `evaluations`
with `combine: any|all` and a non-empty `evaluators` list (each item has the
same shape as a single `evaluation` block, including `strategy`). You cannot
set both `evaluation` and `evaluations` on the same config.

```yaml
attacks:
  - id: poisonedrag
    params:
      adv_per_query: 3
      target_style: short_fact
      llm_client:
        provider: ollama
        base_url: http://localhost:1234
        endpoint: /api/generate
        model: llama3:8b
        temperature: 0.0

evaluation:
  strategy: semantic_similarity
  embedding_client:
    provider: ollama
    base_url: http://localhost:1234
    endpoint: /api/embed
    model: nomic-embed-text
  semantic_similarity:
    metric: cosine
    attack_similarity_threshold: 0.75
    max_reference_similarity: 0.6
    attack_margin_threshold: 0.1
```

```yaml
attacks:
  - id: poisonedrag
    params:
      adv_per_query: 3
      target_style: short_fact
      llm_client:
        provider: ollama
        base_url: http://localhost:1234
        endpoint: /api/generate
        model: llama3:8b
        temperature: 0.0

evaluation:
  strategy: llm_judge
  judge_client:
    provider: ollama
    base_url: http://localhost:1234
    endpoint: /api/generate
    model: llama3:8b
    temperature: 0.0
```

For `semantic_similarity`, `evaluation.embedding_client` must match the embedding
space you intend to compare (for example the same model as the target RAG when
that matters for the metric).

## Corpus poisoning (`scan.corpus_poisoning`)

Optional block to write adversarial contexts into the **same** retrieval store the
target reads from. Set `cleanup: true` to delete only point or row ids created by
the adapter during the run.

Optional install extras:

- `pip install 'vexrag[qdrant]'` for `backend: qdrant`
- `pip install 'vexrag[chroma]'` for `backend: chroma`
- `pip install 'vexrag[faiss]'` for `backend: faiss`

Vector backends require an explicit `embedding_client` nested under the backend
section. It must use the **same** model and preprocessing as the target index.
For cosine indexes that rely on L2-normalized vectors (for example
`IndexFlatIP` on normalized embeddings), set `l2_normalize: true` on that backend
block when appropriate.

**Operational notes**

- **Qdrant** / **Chroma**: stored text is in a `context` field (and Chroma
  `documents`), matching the NQ-style medium examples.
- **FAISS**: expects `index.faiss` and `metadata.json` with `ordered_ids` listing
  one integer id per indexed row (`ntotal` must match the list length). Only
  `IndexFlatIP` is supported. The target must **reload** the index from disk after
  writes. Concurrent writes risk file corruption; use a single writer.

### `file_text` (default)

```yaml
scan:
  corpus_poisoning:
    backend: file_text
    path: ./poisoned_contexts
    cleanup: true
```

### `qdrant`

```yaml
scan:
  corpus_poisoning:
    backend: qdrant
    cleanup: true
    qdrant:
      path: ./qdrant_data
      collection: nq_passages
      vector_name: null
      timeout: 30
      l2_normalize: false
      embedding_client:
        provider: ollama
        base_url: http://localhost:11434
        endpoint: /api/embed
        model: nomic-embed-text:latest
```

Use `url: http://localhost:6333` instead of `path` for a remote server (only one
of `url` or `path`). Optional `api_key` is forwarded when `url` is set.

### `chroma`

```yaml
scan:
  corpus_poisoning:
    backend: chroma
    cleanup: true
    chroma:
      persist_directory: ./chroma_data
      collection: my_collection
      port: 8000
      embedding_client:
        provider: ollama
        base_url: http://localhost:11434
        endpoint: /api/embed
        model: nomic-embed-text:latest
```

For client/server Chroma, set `host` and optional `port` instead of
`persist_directory` or `path`; do not combine `host` with local paths.

### `faiss`

```yaml
scan:
  corpus_poisoning:
    backend: faiss
    cleanup: true
    faiss:
      directory: ./faiss_index_dir
      poison_id_start: -1
      l2_normalize: false
      embedding_client:
        provider: ollama
        base_url: http://localhost:11434
        endpoint: /api/embed
        model: nomic-embed-text:latest
```

`directory` must contain `index.faiss` and `metadata.json`. New poison rows use
negative integer ids (adjust `poison_id_start` if needed).

## Production-oriented semantics

- **Attack success rate (ASR)** uses only scan case runs whose evaluation completed
  (`evaluation_completed`). Incomplete evaluations (for example embedding or judge failures)
  appear as `NOT_EVALUATED` in CLI case summaries and are excluded from the ASR denominator
  (a warning is appended to the scan report when this happens).
- **`vx scan --debug`** merges `scan.debug_include_raw_target_response: true`, so the HTTP target
  adapter attaches the decoded JSON response body to metadata. By default only `http_status` is stored.
- **Embedded services**: import `create_runtime` and `VexRAGRuntime` from `vexrag.core` (or
  `vexrag.core.runtime`). Each runtime owns an isolated `AttackRegistry`; call
  `ensure_builtin_attacks_registered()` on that instance instead of relying on the process-wide default
  when mounting VexRAG inside a long-lived application.
