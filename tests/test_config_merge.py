from vexrag.core.config import deep_merge_mappings


def test_deep_merge_nested_dict() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"b": 2, "nested": {"y": 9, "z": 3}}
    got = deep_merge_mappings(base, override)
    assert got == {"a": 1, "b": 2, "nested": {"x": 1, "y": 9, "z": 3}}
