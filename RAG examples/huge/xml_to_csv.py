import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

CSV_COLUMNS = [
    "Id",
    "PostTypeId",
    "ParentId",
    "AcceptedAnswerId",
    "CreationDate",
    "Score",
    "ViewCount",
    "Title",
    "Body",
    "Tags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert StackOverflow Posts.xml to TSV/CSV for later Qdrant ingestion."
    )
    parser.add_argument(
        "--input-xml",
        required=True,
        help="Path to StackOverflow Posts.xml",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV/TSV path",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke conversion",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Progress update interval",
    )
    return parser.parse_args()


def row_payload(attrs: dict[str, str]) -> dict[str, str]:
    return {col: attrs.get(col, "") for col in CSV_COLUMNS}


def convert_xml_to_csv(
    input_xml: Path,
    output_csv: Path,
    limit_rows: int | None,
    progress_every: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=CSV_COLUMNS,
            delimiter="\t",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()

        progress = tqdm(unit="rows", desc="Converting XML->CSV")
        context = ET.iterparse(input_xml, events=("end",))
        for _, elem in context:
            if elem.tag != "row":
                continue

            writer.writerow(row_payload(elem.attrib))
            processed += 1
            progress.update(1)

            if processed % progress_every == 0:
                progress.set_postfix({"written": processed})

            # Free parsed XML subtree memory
            elem.clear()

            if limit_rows is not None and processed >= limit_rows:
                break

        progress.close()

    print(f"Finished. Wrote {processed} rows to: {output_csv}")


def main() -> None:
    args = parse_args()
    convert_xml_to_csv(
        input_xml=Path(args.input_xml),
        output_csv=Path(args.output_csv),
        limit_rows=args.limit_rows,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
