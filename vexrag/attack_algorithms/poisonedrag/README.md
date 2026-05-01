# PoisonedRAG

Based on **PoisonedRAG** (*Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models*, USENIX Security 2025); this VexRAG module starts from that work and will be extended here.

- **Paper / official code:** [github.com/sleeepeer/PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)
- **arXiv:** [2402.07867](https://arxiv.org/abs/2402.07867)

## Poisoning Styles

PoisonedRAG supports three generation styles via `attack.poisonedrag.poisoning_style`.

- `original` (default): keep generated adversarial texts as returned by the model, with no extra post-processing.
- `aggressive`: force-add `target_incorrect_answer` to every adversarial text when it is missing.
- `soft`: add a lightweight claim hint (keyword-level cue) instead of a full verbatim claim.

Use `original` for baseline behavior and comparability; use `aggressive`/`soft` only when you intentionally want stronger injection pressure.

## Automatic Case Generation

Generate a ready-to-use YAML file with `cases:` entries (`id`, `query`, `correct_answer`, `target_incorrect_answer`) and plug it into `attack.poisonedrag.case_files`:

```bash
vx generate-cases \
  --config "RAG examples/small/rag_01_in_memory_en/scan_configs_examples/vexrag-poisonedrag-llm-judge-vllm-gemma-3-27b-it-original.yaml" \
  --output "RAG examples/small/rag_01_in_memory_en/scan_configs_examples/cases/poisonedrag-cases-auto.yaml" \
  --count 8 \
  --topic "enterprise RAG security and governance" \
  --overwrite
```
