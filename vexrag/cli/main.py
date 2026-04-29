import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from vexrag.cli.errors import CLIConfigError
from vexrag.cli.scan_builder import build_scan_command

LOGGER = logging.getLogger("vexrag.cli")
FIELD_TEXT_LIMIT = 2_000
CONTEXT_TEXT_LIMIT = 4_000


class ScanCaseReportProtocol(Protocol):
    query: str
    adversarial_text: str
    expected_incorrect_answer: str
    system_response: Any
    evaluation: Any
    case_id: str | None
    run_index: int

    @property
    def successful(self) -> bool: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...


class ScanVerdictProtocol(Protocol):
    value: str


class ScanReportProtocol(Protocol):
    verdict: ScanVerdictProtocol
    cases: tuple[ScanCaseReportProtocol, ...]
    warnings: tuple[str, ...]

    @property
    def success_rate(self) -> float: ...

    @property
    def successful_cases(self) -> int: ...

    @property
    def total_cases(self) -> int: ...


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (CLIConfigError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vx",
        description="VexRAG command-line scanner.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    scan = subcommands.add_parser(
        "scan",
        help="Run a configured scan.",
    )
    _add_config_argument(scan)
    scan.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print the final scan report.",
    )
    scan.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs and raw judge responses.",
    )
    scan.set_defaults(handler=_run_scan)

    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML scan config.",
    )


def _run_scan(args: argparse.Namespace) -> int:
    _configure_logging(quiet=args.quiet, debug=args.debug)
    LOGGER.info("Loading scan config: %s", args.config)
    config = _load_config(args.config)
    _log_config_summary(config)
    LOGGER.info("Building scan command")
    command = build_scan_command(config, base_dir=args.config.parent)
    requests = getattr(command, "requests", ())
    if requests:
        LOGGER.info("Loaded %d scan case(s)", len(requests))
    LOGGER.info("Running scan")
    report = command.run()
    LOGGER.info(
        "Scan complete: verdict=%s, success_rate=%.2f%% (%d/%d)",
        report.verdict.value.upper(),
        report.success_rate * 100,
        report.successful_cases,
        report.total_cases,
    )
    _print_report(report, show_raw_responses=args.debug)
    return 0


def _configure_logging(*, quiet: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _log_config_summary(config: Mapping[str, Any]) -> None:
    target_http = _target_http_config(config)
    if target_http:
        base_url = target_http.get("base_url", "")
        route = target_http.get("route", "")
        method = target_http.get("method", "POST")
        LOGGER.info("Target: %s %s%s", method, base_url, route)

    attack = config.get("attack")
    if isinstance(attack, Mapping):
        methods = [name for name, value in attack.items() if isinstance(value, Mapping)]
        if methods:
            LOGGER.info("Attack: %s", methods[0])

    evaluation = config.get("evaluation")
    if isinstance(evaluation, Mapping):
        strategy = evaluation.get("strategy", "semantic_similarity")
        LOGGER.info("Evaluation: %s", strategy)


def _target_http_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    target = config.get("target_system")
    if not isinstance(target, Mapping):
        return None
    http_config = target.get("http", target)
    if not isinstance(http_config, Mapping):
        return None
    return http_config


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        raw_config = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIConfigError(f"could not read config file: {path}") from exc

    try:
        loaded = _load_yaml(raw_config)
    except ValueError as exc:
        raise CLIConfigError(f"could not parse config file: {path}") from exc

    if not isinstance(loaded, Mapping):
        raise CLIConfigError("config file must contain a mapping")
    return loaded


def _load_yaml(raw_config: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise CLIConfigError("PyYAML is required to read YAML config files") from exc
    try:
        return yaml.safe_load(raw_config)
    except yaml.YAMLError as exc:
        raise CLIConfigError("YAML config file is invalid") from exc


def _print_report(
    report: ScanReportProtocol,
    *,
    show_raw_responses: bool = False,
) -> None:
    print("VexRAG Scan")
    print(f"Verdict: {report.verdict.value.upper()}")
    print(
        "Success rate: "
        f"{report.success_rate:.2%} "
        f"({report.successful_cases}/{report.total_cases})"
    )
    print(f"Cases: {report.total_cases}")
    _print_case_details(report.cases, show_raw_responses=show_raw_responses)

    warnings = _collect_warnings(report)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def _print_case_details(
    cases: Sequence[ScanCaseReportProtocol],
    *,
    show_raw_responses: bool = False,
) -> None:
    if not cases:
        return

    print("Case details:")
    for case_index, case in enumerate(cases, start=1):
        label = case.case_id or f"#{case_index}"
        verdict = "SUCCESS" if case.successful else "FAILED"
        print(f"- Case {case_index}: {label} run {case.run_index} - {verdict}")
        _print_field("Query", case.query)
        _print_field("Expected incorrect answer", case.expected_incorrect_answer)
        _print_field("LLM answer", case.system_response.answer)
        _print_field("Evaluation", _format_evaluation(case.evaluation))
        if case.evaluation.reason:
            _print_field("Evaluation reason", case.evaluation.reason)
        if show_raw_responses and case.evaluation.raw_response is not None:
            _print_field(
                "Judge raw response",
                _format_raw_response(case.evaluation.raw_response),
            )
        _print_contexts(case.system_response.contexts)
        _print_field(
            "Generated adversarial context",
            case.adversarial_text,
            limit=CONTEXT_TEXT_LIMIT,
        )


def _format_evaluation(evaluation: Any) -> str:
    parts = [str(evaluation.strategy)]
    if evaluation.scores:
        scores = ", ".join(
            f"{name}={score:.4f}" for name, score in evaluation.scores.items()
        )
        parts.append(scores)
    judge_answer_label = _judge_answer_label(evaluation)
    if judge_answer_label:
        parts.append(f"judge_answer_label={judge_answer_label}")
    return "; ".join(parts)


def _judge_answer_label(evaluation: Any) -> object | None:
    metadata = getattr(evaluation, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("judge_answer_label")
    return None


def _format_raw_response(raw_response: object) -> str:
    if isinstance(raw_response, Mapping):
        return json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    return str(raw_response)


def _print_contexts(contexts: Sequence[str]) -> None:
    print(f"  Retrieved contexts: {len(contexts)}")
    for context_index, context in enumerate(contexts, start=1):
        _print_field(
            f"Context {context_index}",
            context,
            indent="    ",
            limit=CONTEXT_TEXT_LIMIT,
        )


def _print_field(
    label: str,
    value: object,
    *,
    indent: str = "  ",
    limit: int = FIELD_TEXT_LIMIT,
) -> None:
    text = _truncate_text(str(value).strip(), limit=limit)
    if "\n" not in text and len(text) <= 120:
        print(f"{indent}{label}: {text}")
        return

    print(f"{indent}{label}:")
    for line in text.splitlines() or [""]:
        print(f"{indent}  {line}")


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}... [truncated {omitted} chars]"


def _collect_warnings(report: ScanReportProtocol) -> tuple[str, ...]:
    warnings: list[str] = []
    warnings.extend(report.warnings)
    for case in report.cases:
        warnings.extend(case.warnings)
    return tuple(dict.fromkeys(warnings))


if __name__ == "__main__":
    raise SystemExit(main())
