# VexRAG

![Project: in development](https://img.shields.io/badge/project-in%20development-F59E0B?style=for-the-badge)

A toolkit for assessing the **functional correctness** of retrieval-augmented generation (RAG) systems under attack conditions.

**Sample RAG stacks** for getting started: [RAG examples](RAG%20examples/README.md).

## TODO

Canonical checklist: [notes/TODO.md](notes/TODO.md).

## Done
- [x] Small RAG (in-memory)
- [x] PoisonedRAG target scan pipeline with core target, scan, and evaluation contracts
- [x] Core package facade exports clarified for shared APIs

## In Progress
- [ ] PoisonedRAG through Core and CLI: configurable HTTP target adapter, YAML-driven scan command, evaluation strategy wiring, and `rag_01_in_memory_en` usage example
- [ ] Medium RAG with Chroma DB

## Next
- [ ] Large RAG with Huge Dataset (StackOverflow + Qdrant)

## Ideas / Backlog
- [ ] Red-team testing methods for API-interacting RAG services (local RAG targets)
- [ ] Red-team testing methods for the VexRAG CLI (local RAG targets)
