# VexRAG

![Project: in development](https://img.shields.io/badge/project-in%20development-F59E0B?style=for-the-badge)

A toolkit for assessing the **functional correctness** of retrieval-augmented generation (RAG) systems under attack conditions.

**Sample RAG stacks** for getting started: [RAG examples](RAG%20examples/README.md).

## TODO

Canonical checklist: [notes/TODO.md](notes/TODO.md).

## Done
- [x] Small RAG (in-memory)
- [x] PoisonedRAG target scan pipeline with core target, scan, and evaluation contracts
- [x] PoisonedRAG CLI scan flow wired from YAML config with multi-context poisoning runs
- [x] Core package facade exports clarified for shared APIs
- [x] StackOverflow XML/TSV to Qdrant ingestion scripts for large dataset indexing

## In Progress
- [ ] PoisonedRAG hardening: broaden scenario coverage, stabilize metrics, and add end-to-end validation runs
- [ ] Medium RAG with Chroma DB

## Next
- [ ] Wire the huge StackOverflow + Qdrant example into a full end-to-end runnable RAG demo

## Ideas / Backlog
- [ ] Red-team testing methods for API-interacting RAG services (local RAG targets)
- [ ] Red-team testing methods for the VexRAG CLI (local RAG targets)
