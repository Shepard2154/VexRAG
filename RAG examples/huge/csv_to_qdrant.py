import argparse
import csv
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class PostRecord:
    post_id: int
    post_type_id: int
    parent_id: int | None
    score: int | None
    title: str
    body: str
    tags: str

    @property
    def text_for_embedding(self) -> str:
        if self.post_type_id == 1:
            return f"Q: {self.title}\n{self.body}".strip()
        return f"A: {self.body}".strip()

    @property
    def payload(self) -> dict[str, object]:
        item_type = "question" if self.post_type_id == 1 else "answer"
        payload: dict[str, object] = {
            "type": item_type,
            "post_id": self.post_id,
            "post_type_id": self.post_type_id,
            "score": self.score,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
        }
        if self.parent_id is not None:
            payload["parent_id"] = self.parent_id
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load StackOverflow CSV/TSV into Qdrant with embeddings."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--collection", default="stackoverflow_rag")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--skip-create-collection",
        action="store_true",
        help="Do not recreate collection before upsert.",
    )
    return parser.parse_args()


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def iter_posts(path: str, limit_rows: int | None) -> list[PostRecord]:
    records: list[PostRecord] = []
    with open(path, encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        for idx, row in enumerate(reader):
            if limit_rows is not None and idx >= limit_rows:
                break

            post_type_id = to_int(row.get("PostTypeId"))
            post_id = to_int(row.get("Id"))
            if post_type_id not in (1, 2) or post_id is None:
                continue

            body = (row.get("Body") or "").strip()
            title = (row.get("Title") or "").strip()
            if not body and not title:
                continue

            records.append(
                PostRecord(
                    post_id=post_id,
                    post_type_id=post_type_id,
                    parent_id=to_int(row.get("ParentId")),
                    score=to_int(row.get("Score")),
                    title=title,
                    body=body,
                    tags=(row.get("Tags") or "").strip(),
                )
            )
    return records


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    skip_create: bool,
) -> None:
    if skip_create:
        return
    if client.collection_exists(collection):
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_records(
    client: QdrantClient,
    collection: str,
    model: SentenceTransformer,
    records: list[PostRecord],
    batch_size: int,
) -> None:
    for start in tqdm(range(0, len(records), batch_size), desc="Uploading batches"):
        batch = records[start : start + batch_size]
        texts = [item.text_for_embedding for item in batch]
        vectors = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload=batch[i].payload,
            )
            for i in range(len(batch))
        ]
        client.upsert(collection_name=collection, points=points)


def main() -> None:
    args = parse_args()
    model = SentenceTransformer(args.embedding_model, device=args.device)
    client = QdrantClient(url=args.qdrant_url)

    records = iter_posts(args.input_csv, args.limit_rows)
    if not records:
        raise RuntimeError("No suitable posts found in CSV.")

    vector_size = len(model.encode(["size probe"], show_progress_bar=False)[0])
    ensure_collection(
        client=client,
        collection=args.collection,
        vector_size=vector_size,
        skip_create=args.skip_create_collection,
    )

    upsert_records(
        client=client,
        collection=args.collection,
        model=model,
        records=records,
        batch_size=args.batch_size,
    )
    print(f"Done. Inserted {len(records)} records into '{args.collection}'.")


if __name__ == "__main__":
    main()
