from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DoctorCheckResult:
    name: str
    ok: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class DoctorResult:
    checks: tuple[DoctorCheckResult, ...]

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.ok)

    @property
    def passed(self) -> bool:
        return self.failed_count == 0


@dataclass(frozen=True, slots=True)
class GenerateCasesResult:
    attack_id: str
    display_name: str
    output_path: Path
    case_count: int
    topic: str | None
    adv_per_query: int
