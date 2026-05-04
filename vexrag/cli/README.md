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
