import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel


@dataclass(frozen=True)
class BenchmarkExample:
    sample_id: int
    question: str
    answer: str
    context: str


class LLMClient:
    """LLM client with Ollama interface."""

    def __init__(self, config: dict) -> None:
        self.timeout_seconds = int(config.get("timeout_seconds", 30))
        self.ollama = config.get("ollama", {})

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return (
            "You are a strict RAG assistant.\n"
            "Rules:\n"
            "1) Answer only from the provided context.\n"
            "2) If the answer is not explicitly supported by context, respond exactly: I don't know\n"
            "3) Do not invent facts.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{joined_context}\n\n"
            "Answer:"
        )

    def generate(self, question: str, contexts: list[str]) -> str:
        return self._generate_ollama(question, contexts)

    def _generate_ollama(self, question: str, contexts: list[str]) -> str:
        base_url = str(self.ollama.get("base_url", "http://localhost:11434")).rstrip("/")
        model = self.ollama.get("model")
        if not model:
            raise RuntimeError("llm.ollama.model is required in config.json")
        payload = {
            "model": str(model),
            "prompt": self._build_prompt(question, contexts),
            "stream": False,
            "options": {"temperature": float(self.ollama.get("temperature", 0.0))},
        }
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

class SmallRAG:
    """Tiny in-memory RAG with benchmark-like QA data."""

    def __init__(
        self,
        examples: list[BenchmarkExample],
        model_name: str,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.examples = examples
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.llm_client = llm_client
        self.contexts = [example.context for example in examples]
        self.context_vectors = self._embed_documents(self.contexts)

    def _embed_documents(self, documents: list[str]) -> np.ndarray:
        vectors = np.array(self.embedding.embed_documents(documents), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _embed_query(self, query: str) -> np.ndarray:
        vector = np.array(self.embedding.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple[BenchmarkExample, float]]:
        query_vec = self._embed_query(query)
        scores = self.context_vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.examples[idx], float(scores[idx])) for idx in top_indices]

    def answer_with_contexts(self, query: str, top_k: int = 2) -> tuple[str, list[str]]:
        if not self.llm_client:
            raise RuntimeError("LLM client is not configured.")
        retrieved = self.retrieve(query, top_k=top_k)
        retrieved_examples = [item for item, _score in retrieved]
        context_texts = [item.context for item in retrieved_examples]
        answer_text = self.llm_client.generate(
            question=query,
            contexts=context_texts,
        )
        return answer_text, context_texts

    def answer(self, query: str, top_k: int = 2) -> str:
        text, _ = self.answer_with_contexts(query, top_k=top_k)
        return text


class RAGRequest(BaseModel):
    query: str


def load_examples(path: Path, contexts_dir: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = int(row["id"])
            context_path = contexts_dir / f"{sample_id}.txt"
            context = context_path.read_text(encoding="utf-8").strip()
            examples.append(
                BenchmarkExample(
                    sample_id=sample_id,
                    question=row["question"],
                    answer=row["answer"],
                    context=context,
                )
            )
    return examples


def create_app(rag: SmallRAG, top_k: int) -> FastAPI:
    app = FastAPI(title="Small In-Memory RAG")

    @app.post("/model/with-context")
    def model_with_context(request: RAGRequest) -> dict[str, str | list[str]]:
        try:
            answer_text, context_texts = rag.answer_with_contexts(
                request.query, top_k=top_k
            )
            return {"answer": answer_text, "contexts": context_texts}
        except (requests.RequestException, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def load_config(config_path: Path) -> dict:
    defaults = {
        "dataset": "benchmark.jsonl",
        "contexts_dir": "contexts",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 2,
        "host": "0.0.0.0",
        "port": 9003,
        "llm": {
            "timeout_seconds": 30,
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3:8b",
                "temperature": 0.0,
            },
        },
    }
    if not config_path.exists():
        return defaults

    with config_path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    return {**defaults, **loaded}


BASE_DIR = Path(__file__).parent
CONFIG = load_config(BASE_DIR / "config.json")
if os.environ.get("OLLAMA_BASE_URL"):
    llm_cfg = CONFIG.setdefault("llm", {})
    ollama_cfg = llm_cfg.setdefault("ollama", {})
    ollama_cfg["base_url"] = str(os.environ["OLLAMA_BASE_URL"]).rstrip("/")
DEFAULT_DATASET = BASE_DIR / str(CONFIG["dataset"])
DEFAULT_CONTEXTS_DIR = BASE_DIR / str(CONFIG["contexts_dir"])
DEFAULT_EMBEDDING_MODEL = str(CONFIG["embedding_model"])
DEFAULT_TOP_K = int(CONFIG["top_k"])
DEFAULT_HOST = str(CONFIG["host"])
DEFAULT_PORT = int(CONFIG["port"])
LLM_CLIENT = LLMClient(CONFIG.get("llm", {}))

_examples = load_examples(DEFAULT_DATASET, DEFAULT_CONTEXTS_DIR)
_rag = SmallRAG(
    _examples,
    model_name=DEFAULT_EMBEDDING_MODEL,
    llm_client=LLM_CLIENT,
)
app = create_app(_rag, top_k=DEFAULT_TOP_K)


if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
