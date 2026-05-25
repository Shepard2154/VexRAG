import hashlib


def stable_generated_case_id(seed_text: str, *, prefix: str = "generated_case") -> str:
    digest = hashlib.sha256(seed_text.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"
