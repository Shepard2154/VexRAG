from dataclasses import dataclass

from vexrag.core.attack_configurator.registry import AttackMethodRegistry
from vexrag.core.evaluation.registry import EvaluationRegistry
from vexrag.core.llm.providers.defaults import create_default_llm_provider_registry
from vexrag.core.llm.providers.registry import LLMProviderRegistry
from vexrag.core.retrieval.registry import RetrievalBackendRegistry
from vexrag.core.target_systems.registry import TargetSystemRegistry


@dataclass(frozen=True, slots=True)
class ScanRegistries:
    llm_providers: LLMProviderRegistry
    retrieval_backends: RetrievalBackendRegistry
    target_systems: TargetSystemRegistry
    evaluations: EvaluationRegistry
    attack_methods: AttackMethodRegistry


def create_default_scan_registries(
    *,
    retrieval_backends: RetrievalBackendRegistry,
    target_systems: TargetSystemRegistry,
    evaluations: EvaluationRegistry,
    attack_methods: AttackMethodRegistry,
) -> ScanRegistries:
    return ScanRegistries(
        llm_providers=create_default_llm_provider_registry(),
        retrieval_backends=retrieval_backends,
        target_systems=target_systems,
        evaluations=evaluations,
        attack_methods=attack_methods,
    )
