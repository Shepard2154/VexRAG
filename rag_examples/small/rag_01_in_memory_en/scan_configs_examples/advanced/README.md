# Advanced scan configs

Not required for first-time setup. Use [`../ollama-default.yaml`](../ollama-default.yaml) unless you explicitly need these integrations.

| File | Requires |
|------|----------|
| `hijackrag-llm-judge-ollama.yaml` | Ollama `llama3:8b` |
| `hijackrag-semantic-similarity-ollama.yaml` | Ollama `llama3:8b`, `nomic-embed-text:latest` |
| `chain-qwen25-14b.yaml` | Ollama `qwen2.5:14b`, `nomic-embed-text:latest` |
| `poisonedrag-llm-judge-vllm-gemma.yaml` | vLLM `google/gemma-3-27b-it` at `http://localhost:8017/v1` |
| `poisonedrag-semantic-similarity-vllm-gemma.yaml` | vLLM Gemma + Ollama `nomic-embed-text:latest` |

Run from the example directory, e.g.:

```bash
vx scan --config scan_configs_examples/advanced/hijackrag-llm-judge-ollama.yaml
```
