import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_NQ_DATASET = "google-research-datasets/natural_questions"


@dataclass(frozen=True)
class BenchmarkExample:
    sample_id: int
    question: str
    answer: str
    context: str


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(part.strip() for part in self._parts if part.strip())


def _strip_html(text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.get_text()


def normalize_context(text: str, max_chars: int = 4096) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars]
    if " " in truncated:
        return truncated.rsplit(" ", 1)[0]
    return truncated


def corpus_fingerprint(examples: list[BenchmarkExample]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.sample_id):
        digest.update(f"{example.sample_id}\0{example.context}\0".encode())
    return digest.hexdigest()


def extract_question(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, str):
        return question.strip()
    if isinstance(question, Mapping):
        text = question.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def extract_context(row: Mapping[str, Any]) -> str:
    for key in ("context", "document_text", "passage", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_context(value.strip())

    document = row.get("document")
    if isinstance(document, Mapping):
        tokens = document.get("tokens")
        if isinstance(tokens, Mapping):
            token_values = tokens.get("token")
            if isinstance(token_values, list):
                joined = " ".join(
                    str(token).strip() for token in token_values if str(token).strip()
                )
                if joined:
                    return normalize_context(joined)

        for key in ("text", "document_text", "html"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                plain = _strip_html(value) if key == "html" else value.strip()
                if plain:
                    return normalize_context(plain)
    return ""


def extract_answer(row: Mapping[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, list):
        for item in answer:
            if isinstance(item, str) and item.strip():
                return item.strip()
    annotations = row.get("annotations")
    if isinstance(annotations, Mapping):
        short_answers = annotations.get("short_answers")
        if isinstance(short_answers, list):
            for item in short_answers:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    return ""


def load_examples(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = int(row["id"])
            context = str(row.get("context", "")).strip()
            if not context:
                continue
            examples.append(
                BenchmarkExample(
                    sample_id=sample_id,
                    question=row["question"],
                    answer=row["answer"],
                    context=context,
                )
            )
    return examples


def load_nq_examples(config: Mapping[str, Any]) -> list[BenchmarkExample]:
    dataset_name = str(config.get("name", DEFAULT_NQ_DATASET)).strip()
    split = str(config.get("split", "train")).strip()
    try:
        limit = int(config.get("limit", 2000))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("nq_dataset.limit must be an integer") from exc
    if limit <= 0:
        raise RuntimeError("nq_dataset.limit must be greater than 0")
    try:
        max_context_chars = int(config.get("max_context_chars", 4096))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("nq_dataset.max_context_chars must be an integer") from exc
    if max_context_chars <= 0:
        raise RuntimeError("nq_dataset.max_context_chars must be greater than 0")
    cache_dir_raw = config.get("cache_dir")
    cache_dir = str(cache_dir_raw).strip() if isinstance(cache_dir_raw, str) else None

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        cache_dir=cache_dir or None,
    )
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(dataset, start=1):
        question = extract_question(row)
        context = extract_context(row)
        answer = extract_answer(row)
        if not question or not context:
            continue
        examples.append(
            BenchmarkExample(
                sample_id=index,
                question=question,
                answer=answer,
                context=normalize_context(context, max_chars=max_context_chars),
            )
        )
        if len(examples) >= limit:
            break
    if not examples:
        raise RuntimeError("could not load NQ examples with question+context")
    return examples
