from vexrag.core.runtime import create_runtime


def test_create_runtime_keeps_registry_isolated() -> None:
    r1 = create_runtime()
    r2 = create_runtime()

    assert r1.registry is not r2.registry
    assert not r2.registry.ids()
    r1.ensure_builtin_attacks_registered()
    assert r1.registry.ids()
    assert not r2.registry.ids()
