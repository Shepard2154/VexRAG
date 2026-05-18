from collections.abc import Sequence


def nonempty_stripped_texts(texts: Sequence[str]) -> list[str]:
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]
