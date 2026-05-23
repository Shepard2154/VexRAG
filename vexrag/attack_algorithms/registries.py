from pathlib import Path

from vexrag.core.attack_configurator.registry import (
    AttackMethodRegistry,
    AttackMethodRegistryBuilder,
)
from vexrag.core.llm.providers.defaults import create_default_llm_provider_registry
from vexrag.core.scan.builder import (
    create_default_evaluation_registry,
    create_default_retrieval_backend_registry,
    create_default_target_system_registry,
)
from vexrag.core.scan.builder.registries import (
    ScanRegistries,
    create_default_scan_registries,
)


def create_attack_method_registry() -> AttackMethodRegistry:
    from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
    from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN

    builder = AttackMethodRegistryBuilder()
    builder.register(HIJACK_PLUGIN)
    builder.register(POISON_PLUGIN)
    return builder.build()


def create_scan_registries(*, base_dir: Path | None = None) -> ScanRegistries:
    llm_providers = create_default_llm_provider_registry()
    return create_default_scan_registries(
        retrieval_backends=create_default_retrieval_backend_registry(
            llm_providers=llm_providers,
            base_dir=base_dir,
        ),
        target_systems=create_default_target_system_registry(),
        evaluations=create_default_evaluation_registry(),
        attack_methods=create_attack_method_registry(),
    )
