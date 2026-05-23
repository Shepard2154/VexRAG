from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vexrag.core.evaluation.contracts import Evaluator
from vexrag.core.scan.config.errors import EvaluationConfigError

EvaluatorBuilder = Callable[..., Evaluator]
SimilarityMetric = Callable[..., float]


@dataclass(frozen=True, slots=True)
class EvaluationRegistry:
    evaluator_builders: Mapping[str, EvaluatorBuilder]
    metrics: Mapping[str, SimilarityMetric]

    def get_evaluator_builder(self, strategy: str) -> EvaluatorBuilder:
        key = strategy.strip()
        try:
            return self.evaluator_builders[key]
        except KeyError as err:
            supported = ", ".join(sorted(self.evaluator_builders))
            raise EvaluationConfigError(
                f"evaluation.strategy must be one of: {supported}"
            ) from err

    def get_metric(self, metric: str) -> SimilarityMetric:
        key = metric.strip()
        try:
            return self.metrics[key]
        except KeyError as err:
            supported = ", ".join(sorted(self.metrics))
            raise EvaluationConfigError(
                f"evaluation.embedding_similarity.metric must be one of: {supported}"
            ) from err
