from collections.abc import Sequence


def nonempty_stripped_strs(texts: Sequence[str]) -> list[str]:
    return [t.strip() for t in texts if isinstance(t, str) and t.strip()]
