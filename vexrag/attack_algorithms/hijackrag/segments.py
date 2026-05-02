import json
import re
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


def default_hijack_segments_path() -> Path:
    return Path(__file__).resolve().parent / "hijack_segment.json"


def load_hijack_segments(path: Path) -> tuple[HijackSegmentRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"hijack segment dataset not found: {path}")

    raw = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[HijackSegmentRecord] = []
    index = 0
    length = len(raw)
    while index < length:
        while index < length and raw[index] in " \t\r\n":
            index += 1
        if index >= length:
            break
        try:
            row, end = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            next_brace = raw.find("{", index + 1)
            if next_brace == -1:
                break
            index = next_brace
            continue
        index = end
        if not isinstance(row, dict):
            continue
        try:
            segment_id = str(row["id"]).strip()
            template = str(row["hijack_segment"])
        except (KeyError, TypeError):
            continue
        if not segment_id or not template.strip():
            continue
        records.append(HijackSegmentRecord(segment_id=segment_id, template=template))
    if not records:
        raise ValueError(f"No hijack segments parsed from {path}")
    return tuple(records)


def apply_hijack_insert(template: str, hijack_insert: str) -> str:
    if not INSERT_PLACEHOLDER_PATTERN.search(template):
        return f"{template.rstrip()}\n{hijack_insert}".strip()
    return INSERT_PLACEHOLDER_PATTERN.sub(hijack_insert, template).strip()
