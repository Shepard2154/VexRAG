from vexrag.core.llm.providers.errors import ProviderServiceError
from vexrag.core.scan.config.errors import ScanConfigError


class UseCaseError(Exception):
    """Base error for CLI use cases with a stable exit code."""

    exit_code: int = 4

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UseCaseConfigError(UseCaseError, ScanConfigError):
    """Invalid YAML, flags, or scan configuration."""

    exit_code = 2


class UseCaseDependencyError(UseCaseError):
    """Required external service (target API, Ollama, LLM) is unavailable."""

    exit_code = 3


def exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, UseCaseError):
        return exc.exit_code
    if isinstance(exc, ProviderServiceError):
        return UseCaseDependencyError.exit_code
    if isinstance(exc, (ScanConfigError, ValueError)):
        return UseCaseConfigError.exit_code
    if isinstance(exc, RuntimeError):
        return 4
    return 4
