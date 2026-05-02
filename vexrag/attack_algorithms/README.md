# Attack algorithms

Reference implementations of peer-reviewed and widely cited attack techniques for evaluating RAG security.

## Adding a new attack

1. Create a package under `vexrag/attack_algorithms/<attack_id>/` (domain modules as needed).
2. Add `plugin.py` implementing `vexrag.core.attacks.plugin.AttackPlugin` and a `register(registry)` function that calls `registry.register(...)`.
3. Register the package from `vexrag.core.attacks.builtins.register_builtin_attacks` (import your `plugin.register` there).
4. Add a minimal YAML example under `RAG examples/` or your own config tree with `attack.<attack_id>:`.

Third-party packages can also expose `[project.entry-points."vexrag.attacks"]` pointing at a callable `register(registry)`.
