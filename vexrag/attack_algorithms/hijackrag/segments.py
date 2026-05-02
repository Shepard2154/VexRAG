import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

INSERT_PLACEHOLDER_PATTERN = re.compile(
    re.escape("[INSERT PROMPT HERE]"),
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class HijackSegmentRecord:
    segment_id: str
    template: str


_SEGMENT_TEMPLATES_DIR = Path(__file__).resolve().parent / "segment_templates"
DEFAULT_HIJACK_SEGMENT_TEMPLATE_FILENAME = "default_en.jsonl"


def default_hijack_segments_path() -> Path:
    """Built-in English segment templates (paper-style HijackRAG inserts)."""
    return _SEGMENT_TEMPLATES_DIR / DEFAULT_HIJACK_SEGMENT_TEMPLATE_FILENAME


def load_hijack_segments(path: Path) -> tuple[HijackSegmentRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"hijack segment dataset not found: {path}")

    raw = path.read_text(encoding="utf-8")
    records = tuple(
        record
        for row in _iter_json_objects(raw)
        if (record := _parse_hijack_segment_record(row)) is not None
    )
    if not records:
        raise ValueError(f"No hijack segments parsed from {path}")
    return records


def _iter_json_objects(raw: str) -> Iterator[Mapping[str, object]]:
    decoder = json.JSONDecoder()
    index = 0
    length = len(raw)

    while index < length:
        while index < length and raw[index] in " \t\r\n":
            index += 1
        if index >= length:
            break
        try:
            row, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            index = raw.find("{", index + 1)
            if index == -1:
                break
            continue
        if isinstance(row, dict):
            yield row


def _parse_hijack_segment_record(
    row: Mapping[str, object],
) -> HijackSegmentRecord | None:
    try:
        segment_id = str(row["id"]).strip()
        template = str(row["hijack_segment"])
    except (KeyError, TypeError):
        return None

    if not segment_id or not template.strip():
        return None

    return HijackSegmentRecord(segment_id=segment_id, template=template)


def apply_hijack_insert(template: str, hijack_insert: str) -> str:
    if not INSERT_PLACEHOLDER_PATTERN.search(template):
        return f"{template.rstrip()}\n{hijack_insert}".strip()
    return INSERT_PLACEHOLDER_PATTERN.sub(hijack_insert, template).strip()
