# StackOverflowRAG (Huge)

Operational setup for a large StackOverflow-based RAG index using:
- XML dump (`Posts.xml`)
- CSV intermediate export
- Qdrant vector storage

The full XML → CSV → Qdrant pipeline has **not** been exercised end-to-end in this repository. Errors may occur.

## What This Example Provides

- `xml_to_csv.py`: stream-converts `Posts.xml` into a tab-separated CSV file.
- `csv_to_qdrant.py`: reads the CSV in batches, generates embeddings, and upserts to Qdrant.
- `requirements.txt`: Python dependencies for both scripts.

## Prerequisites

- Python 3.11+
- Running Qdrant
- Enough disk for intermediate CSV and enough RAM/VRAM for embedding batches

Data source used in this setup:
- StackExchange archive: [https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z)

## 1) Install dependencies

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "RAG examples/huge/requirements.txt"
```

## 2) Convert XML to CSV

```bash
python "RAG examples/huge/xml_to_csv.py" \
  --input-xml /path/to/Posts.xml \
  --output-csv /path/to/stackoverflow_posts.tsv
```

Notes:
- Output is tab-separated (`\t`) to reduce quoting issues with large HTML bodies.
- The script streams XML (`iterparse`) and does not load the full dump into memory.

## 3) Start Qdrant

Example with Docker:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 \
  -v /path/to/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

## 4) Load CSV into Qdrant

```bash
python "RAG examples/huge/csv_to_qdrant.py" \
  --input-csv /path/to/stackoverflow_posts.tsv \
  --collection stackoverflow_rag \
  --qdrant-url http://<qdrant-host>:6333 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256
```

Optional flags:
- `--device cuda` (or `cpu`)
- `--limit-rows 10000` for smoke runs
- `--skip-create-collection` to append into an existing collection

## 5) Quick verification

```bash
python "RAG examples/huge/csv_to_qdrant.py" \
  --input-csv /path/to/stackoverflow_posts.tsv \
  --collection stackoverflow_rag \
  --qdrant-url http://<qdrant-host>:6333 \
  --limit-rows 1000 \
  --batch-size 128
```

If this completes successfully, run the full load without `--limit-rows`.
