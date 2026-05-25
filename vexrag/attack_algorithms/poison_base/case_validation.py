import re


def required_text(value: object, *, field: str, error_type: type[Exception]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"Field '{field}' must be a non-empty string.")
    return value.strip()


def normalize_case_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    if not slug:
        return "generated_case"
    return slug[:64]
