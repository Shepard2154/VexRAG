# HijackRAG

VexRAG implementation of **HijackRAG** (*Hijacking Attacks against Retrieval-Augmented Large Language Models*). Segment templates ship as `hijack_segment.json` next to this module.

- **Paper / reference implementation:** [github.com/BarryZYC/HijackRAG](https://github.com/BarryZYC/HijackRAG)
- **arXiv:** [2410.22832](https://arxiv.org/abs/2410.22832)

To draft `cases:` YAML from an existing Hijack scan config (same `attack.hijackrag.llm_client` as a real run), use `vx generate-cases` with `--attack hijackrag` or a config that only defines `attack.hijackrag` (see the small in-memory RAG example README).
