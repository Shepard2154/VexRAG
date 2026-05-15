class EvaluatorError(Exception):
    """Base exception for evaluation-layer."""


class JudgeResponseValidationError(EvaluatorError):
    """Raised when an LLM judge response does not match the expected schema."""
