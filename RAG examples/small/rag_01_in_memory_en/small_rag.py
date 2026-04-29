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
        timeout_seconds = config["timeout_seconds"]
        self.timeout_seconds = None if timeout_seconds is None else int(timeout_seconds)
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise RuntimeError("llm.timeout_seconds must be greater than 0")
        self.ollama = config["ollama"]

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
        base_url = str(self.ollama["base_url"]).rstrip("/")
        model = self.ollama.get("model")
        if not model:
            raise RuntimeError("llm.ollama.model is required in config.json")
        payload = {
            "model": str(model),
            "prompt": self._build_prompt(question, contexts),
            "stream": False,
            "options": {"temperature": float(self.ollama["temperature"])},
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
        extra_contexts_dir: Path | None = None,
    ) -> None:
        self._base_examples = list(examples)
        self.extra_contexts_dir = extra_contexts_dir
        self._extra_context_files: tuple[Path, ...] = ()
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.llm_client = llm_client
        self._set_examples(self._base_examples)

    def _set_examples(self, examples: list[BenchmarkExample]) -> None:
        self.examples = examples
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

    def retrieve(
        self, query: str, top_k: int = 2
    ) -> list[tuple[BenchmarkExample, float]]:
        query_vec = self._embed_query(query)
        scores = self.context_vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.examples[idx], float(scores[idx])) for idx in top_indices]

    def answer_with_contexts(self, query: str, top_k: int = 2) -> tuple[str, list[str]]:
        if not self.llm_client:
            raise RuntimeError("LLM client is not configured.")
        self._reload_extra_contexts()
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

    def _reload_extra_contexts(self) -> None:
        if self.extra_contexts_dir is None:
            return
        context_files = tuple(sorted(self.extra_contexts_dir.glob("poisonedrag_*.txt")))
        if context_files == self._extra_context_files:
            return

        extra_examples: list[BenchmarkExample] = []
        for offset, path in enumerate(context_files, start=1):
            try:
                context = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not context:
                continue
            extra_examples.append(
                BenchmarkExample(
                    sample_id=-offset,
                    question="",
                    answer="",
                    context=context,
                )
            )

        self._extra_context_files = context_files
        self._set_examples([*self._base_examples, *extra_examples])


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

    @app.post("/model/context-based-response")
    def context_based_response(request: RAGRequest) -> dict[str, str | list[str]]:
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
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    return apply_env_overrides(config)


def apply_env_overrides(config: dict) -> dict:
    if ollama_base_url := os.getenv("OLLAMA_BASE_URL"):
        config["llm"]["ollama"]["base_url"] = ollama_base_url
    if ollama_model := os.getenv("OLLAMA_MODEL"):
        config["llm"]["ollama"]["model"] = ollama_model
    if ollama_temperature := os.getenv("OLLAMA_TEMPERATURE"):
        config["llm"]["ollama"]["temperature"] = float(ollama_temperature)
    if llm_timeout := os.getenv("LLM_TIMEOUT_SECONDS"):
        config["llm"]["timeout_seconds"] = int(llm_timeout)
    if host := os.getenv("RAG_HOST"):
        config["host"] = host
    if port := os.getenv("PORT", os.getenv("RAG_PORT")):
        config["port"] = int(port)
    return config


BASE_DIR = Path(__file__).parent
CONFIG = load_config(BASE_DIR / "config.json")
DEFAULT_DATASET = BASE_DIR / str(CONFIG["dataset"])
DEFAULT_CONTEXTS_DIR = BASE_DIR / str(CONFIG["contexts_dir"])
DEFAULT_EMBEDDING_MODEL = str(CONFIG["embedding_model"])
DEFAULT_TOP_K = int(CONFIG["top_k"])
DEFAULT_HOST = str(CONFIG["host"])
DEFAULT_PORT = int(CONFIG["port"])
LLM_CLIENT = LLMClient(CONFIG["llm"])

_examples = load_examples(DEFAULT_DATASET, DEFAULT_CONTEXTS_DIR)
_rag = SmallRAG(
    _examples,
    model_name=DEFAULT_EMBEDDING_MODEL,
    llm_client=LLM_CLIENT,
    extra_contexts_dir=DEFAULT_CONTEXTS_DIR,
)
app = create_app(_rag, top_k=DEFAULT_TOP_K)


if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
