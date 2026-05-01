# VexRAG — TODO

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
