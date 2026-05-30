from dataclasses import dataclass, field

from vexrag.cli.handlers import scan as scan_handler
from vexrag.core.evaluation import EvaluationInput, EvaluationResult
from vexrag.core.target_systems import TargetSystemQuery, TargetSystemResponse


@dataclass(frozen=True, slots=True)
class DummyEvaluation:
    attack_successful: bool
    completed: bool
    strategy: str = "stub"
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    raw: object | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DummySystemResponse:
    answer: str
    contexts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DummyCase:
    query: str
    adversarial_texts: tuple[str, ...]
    expected_incorrect_answer: str
    system_response: DummySystemResponse
    evaluation: DummyEvaluation
    case_id: str | None = None
    run_index: int = 1
    warnings: tuple[str, ...] = ()

    @property
    def successful(self) -> bool:
        return self.evaluation.completed and self.evaluation.attack_successful


@dataclass(frozen=True, slots=True)
class DummyVerdict:
    value: str


@dataclass(frozen=True, slots=True)
class DummyReport:
    verdict: DummyVerdict
    cases: tuple[DummyCase, ...]
    warnings: tuple[str, ...] = ()

    @property
    def successful_cases(self) -> int:
        return sum(case.successful for case in self.cases)

    @property
    def evaluated_cases(self) -> int:
        return sum(1 for case in self.cases if case.evaluation.completed)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def success_rate(self) -> float:
        if self.evaluated_cases == 0:
            return 0.0
        return self.successful_cases / self.evaluated_cases


class DummyCommand:
    requests: tuple[object, ...] = ()

    def run(self, *, on_case_complete=None):  # type: ignore[no-untyped-def]
        case = DummyCase(
            query="query",
            adversarial_texts=("adv",),
            expected_incorrect_answer="wrong",
            system_response=DummySystemResponse(answer="answer", contexts=("ctx",)),
            evaluation=DummyEvaluation(
                attack_successful=True,
                completed=True,
            ),
            case_id="case-1",
        )
        if on_case_complete is not None:
            on_case_complete(case)
        return DummyReport(
            verdict=DummyVerdict("not_vulnerable"),
            cases=(case,),
        )


def stub_scan_dependencies(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(scan_handler, "load_config", lambda _: {})
    monkeypatch.setattr(scan_handler, "preflight_target_system", lambda _: None)
    monkeypatch.setattr(scan_handler, "preflight_ollama_models", lambda _: None)
    monkeypatch.setattr(scan_handler, "preflight_vllm_models", lambda _: None)
    monkeypatch.setattr(
        scan_handler,
        "build_scan_command",
        lambda *_args, **_kwargs: DummyCommand(),
    )
    monkeypatch.setattr(
        scan_handler,
        "run_scan",
        lambda command, *, on_case_complete=None: command.run(
            on_case_complete=on_case_complete
        ),
    )


class FixedEvaluator:
    __slots__ = ("_ok", "_name")

    def __init__(self, name: str, ok: bool) -> None:
        self._name = name
        self._ok = ok

    @property
    def strategy(self) -> str:
        return self._name

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        return EvaluationResult(
            attack_successful=self._ok,
            strategy=self._name,
            scores={"x": 1.0 if self._ok else 0.0},
            reason=f"reason-{self._name}",
        )


class FakeEval:
    def __init__(self) -> None:
        self.last_input: EvaluationInput | None = None

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        self.last_input = evaluation_input
        return EvaluationResult(
            attack_successful=False,
            strategy="stub",
            scores={},
        )


class FakeTarget:
    def __init__(self) -> None:
        self.last_query: TargetSystemQuery | None = None

    def answer(self, request: TargetSystemQuery) -> TargetSystemResponse:
        self.last_query = request
        return TargetSystemResponse(answer="stub answer", contexts=())
