# Attack algorithms

Reference implementations of peer-reviewed and widely cited attack techniques for evaluating RAG security.

## Adding a new attack

1. Create a package under `vexrag/attack_algorithms/<attack_id>/` (domain modules as needed).
2. Add `plugin.py` implementing `vexrag.core.attacks.plugin.AttackPlugin` and a `register(registry)` function that calls `registry.register(...)`.
3. Register the package from `vexrag.core.attacks.builtins.register_builtin_attacks` (import your `plugin.register` there).
4. Add a minimal YAML example under `RAG examples/` or your own config tree with `attacks: [{ id: <attack_id>, params: ... }]`.

Third-party packages can also expose `[project.entry-points."vexrag.attacks"]` pointing at a callable `register(registry)`.

## YAML / config parsing for plugins

Use the stable option parsers in `vexrag.core.config.options` (for example `required_string`, `optional_string`, `int_option`, `bool_option`, `float_option`, `path_option`) when reading attack-specific keys from a loaded config mapping. Use the public builders in `vexrag.core.config.build` (for example `build_target_system`, `build_evaluation_strategy`, `build_corpus_poisoner`, `attack_section`, `attack_llm_client_section`) to assemble core objects. Do not import underscore-prefixed helpers from `config.build`; those are internal to the core.
