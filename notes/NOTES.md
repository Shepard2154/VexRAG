# Notes

Scratch space for design notes.

## PoisonedRAG Strategy Notes (gemma-3-27b-it)

### Strategy names

- `original`: baseline generation, no extra claim injection.
- `aggressive`: force-inject full `target_incorrect_answer` into every generated adversarial text.
- `soft`: inject only a lighter claim hint (keyword cue), not the full verbatim claim.

Default strategy should remain `original`.

### Observed behavior (gemma-3-27b-it, 3-case suite)

With older "banana" target claims (intentionally absurd poisoned answers containing tokens like "bananas" / "banana indexes", used as a synthetic stress-test baseline):
- `llm_judge`: `original` 33.33%, `soft` 66.67%, `aggressive` 100.00%.
- `semantic_similarity`: all three stayed around 33.33%.

With more plausible (no-banana) target claims (domain-realistic but still incorrect shifts, e.g., "external web search" for RAG definition and "continuous synchronization" for vector stores):
- `llm_judge`: all three reached 66.67%.
- `semantic_similarity`: all three reached 66.67%.

Takeaway: claim phrasing quality can dominate style choice; `aggressive` helps most when target claims are weak/noisy.
