import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from vexrag.cli.errors import CLIConfigError
from vexrag.cli.scan_builder import (
    build_hijackrag_case_generator,
    build_poisonedrag_case_generator,
    build_scan_command,
    resolve_generate_cases_attack,
)

LOGGER = logging.getLogger("vexrag.cli")
FIELD_TEXT_LIMIT = 2_000
CONTEXT_TEXT_LIMIT = 4_000


class ScanCaseReportProtocol(Protocol):
    query: str
    adversarial_texts: tuple[str, ...]
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

    generate_cases = subcommands.add_parser(
        "generate-cases",
        help="Generate PoisonedRAG or HijackRAG cases YAML via the scan config LLM.",
    )
    _add_config_argument(generate_cases)
    generate_cases.add_argument(
        "--attack",
        choices=("auto", "poisonedrag", "hijackrag"),
        default="auto",
        help="Which attack block to use from the YAML (default: auto — single attack block).",
    )
    generate_cases.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the generated YAML cases file.",
    )
    generate_cases.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many cases to generate (default: 5).",
    )
    generate_cases.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Optional topic focus for generated cases.",
    )
    generate_cases.add_argument(
        "--target-style",
        choices=("short_fact", "paragraph"),
        default="short_fact",
        help="Style for correct answers (PoisonedRAG: also incorrect answers; HijackRAG: anchor).",
    )
    generate_cases.add_argument(
        "--adv-per-query",
        type=int,
        default=1,
        metavar="N",
        help="HijackRAG only: adv_per_query per case in the YAML (default: 1). Ignored for PoisonedRAG.",
    )
    generate_cases.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic-ish generation.",
    )
    generate_cases.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    generate_cases.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print the output file path.",
    )
    generate_cases.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs.",
    )
    generate_cases.set_defaults(handler=_run_generate_cases)

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


def _run_generate_cases(args: argparse.Namespace) -> int:
    _configure_logging(quiet=args.quiet, debug=args.debug)
    LOGGER.info("Loading generation config: %s", args.config)
    config = _load_config(args.config)
    explicit = None if args.attack == "auto" else args.attack
    attack_kind = resolve_generate_cases_attack(config, explicit=explicit)
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if output_path.exists() and not args.overwrite:
        raise CLIConfigError(
            f"output file already exists: {output_path}. Use --overwrite to replace it."
        )
    if attack_kind == "poisonedrag":
        LOGGER.info("Building PoisonedRAG automatic case generator")
        generator = build_poisonedrag_case_generator(config)
        LOGGER.info(
            "Generating %d PoisonedRAG case(s) [target_style=%s]",
            args.count,
            args.target_style,
        )
        cases = generator.generate_cases(
            count=args.count,
            topic=args.topic,
            target_style=args.target_style,
            seed=args.seed,
        )
        payload = {"cases": [_poisonedrag_case_mapping(case) for case in cases]}
        yaml_attack = "poisonedrag"
    else:
        LOGGER.info("Building HijackRAG automatic case generator")
        generator = build_hijackrag_case_generator(config)
        adv = int(args.adv_per_query)
        if adv < 1:
            raise CLIConfigError("--adv-per-query must be at least 1")
        LOGGER.info(
            "Generating %d HijackRAG case(s) [target_style=%s, adv_per_query=%d]",
            args.count,
            args.target_style,
            adv,
        )
        cases = generator.generate_cases(
            count=args.count,
            topic=args.topic,
            target_style=args.target_style,
            adv_per_query=adv,
            seed=args.seed,
        )
        payload = {"cases": [_hijackrag_case_mapping(case) for case in cases]}
        yaml_attack = "hijackrag"

    _write_yaml(output_path, payload)
    if args.quiet:
        print(output_path)
        return 0

    label = "PoisonedRAG" if yaml_attack == "poisonedrag" else "HijackRAG"
    print(f"Generated {label} cases")
    print(f"Output: {output_path}")
    print(f"Cases: {len(cases)}")
    if args.topic:
        print(f"Topic: {args.topic}")
    print()
    print("Use in config:")
    print(f"  attack.{yaml_attack}.case_files: ['{output_path}']")
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


def _dump_yaml(content: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise CLIConfigError("PyYAML is required to write YAML files") from exc
    return yaml.safe_dump(
        content,
        allow_unicode=True,
        sort_keys=False,
    )


def _write_yaml(path: Path, content: Mapping[str, Any]) -> None:
    text = _dump_yaml(content)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CLIConfigError(f"could not write YAML output file: {path}") from exc


def _poisonedrag_case_mapping(request: Any) -> Mapping[str, str]:
    case_id = str(getattr(request, "case_id", "") or "").strip()
    query = str(getattr(request, "query", "")).strip()
    correct_answer = str(getattr(request, "correct_answer", "") or "").strip()
    target_incorrect_answer = str(
        getattr(request, "target_incorrect_answer", "") or ""
    ).strip()
    if not query or not correct_answer or not target_incorrect_answer:
        raise CLIConfigError("generated case is missing required fields")
    if not case_id:
        case_id = f"generated_case_{abs(hash(query)) % 1_000_000}"
    return {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "target_incorrect_answer": target_incorrect_answer,
    }


def _hijackrag_case_mapping(request: Any) -> Mapping[str, Any]:
    case_id = str(getattr(request, "case_id", "") or "").strip()
    query = str(getattr(request, "query", "") or "").strip()
    correct_answer = str(getattr(request, "correct_answer", "") or "").strip()
    hijack_insert = str(getattr(request, "hijack_insert", "") or "").strip()
    adv_per_query = int(getattr(request, "adv_per_query", 1) or 1)
    if not query or not correct_answer or not hijack_insert:
        raise CLIConfigError("generated HijackRAG case is missing required fields")
    if not case_id:
        case_id = f"generated_case_{abs(hash(query)) % 1_000_000}"
    row: dict[str, Any] = {
        "id": case_id,
        "query": query,
        "correct_answer": correct_answer,
        "hijack_insert": hijack_insert,
        "adv_per_query": adv_per_query,
    }
    seed = getattr(request, "seed", None)
    if seed is not None:
        row["seed"] = int(seed)
    return row


def _print_report(
    report: ScanReportProtocol,
    *,
    show_raw_responses: bool = False,
) -> None:
    print("VexRAG Scan")
    _print_report_summary(report, verdict_label="Verdict", asr_label="Success rate")
    print(f"Cases: {report.total_cases}")
    print()
    _print_case_details(report.cases, show_raw_responses=show_raw_responses)

    warnings = _collect_warnings(report)
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print()
    _print_report_summary(report, verdict_label="Final verdict", asr_label="ASR")


def _print_report_summary(
    report: ScanReportProtocol,
    *,
    verdict_label: str,
    asr_label: str,
) -> None:
    print(f"{verdict_label}: {report.verdict.value.upper()}")
    print(
        f"{asr_label}: "
        f"{report.success_rate:.2%} "
        f"({report.successful_cases}/{report.total_cases})"
    )


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
        poisoned_hits, poisoned_total = _count_poisoned_context_hits(
            case.system_response.contexts,
            case.adversarial_texts,
        )
        print()
        print(f"  Case {case_index}: {label} (run {case.run_index}) - {verdict}")
        _print_case_statistics(
            evaluation=case.evaluation,
            contexts=case.system_response.contexts,
            poisoned_hits=poisoned_hits,
            poisoned_total=poisoned_total,
            adversarial_texts=case.adversarial_texts,
        )
        _print_field("Query", case.query)
        _print_field("Expected incorrect answer", case.expected_incorrect_answer)
        _print_field("LLM answer", case.system_response.answer)
        if case.evaluation.reason:
            _print_field("Evaluation reason", case.evaluation.reason)
        if show_raw_responses and case.evaluation.raw_response is not None:
            _print_field(
                "Judge raw response",
                _format_raw_response(case.evaluation.raw_response),
            )
        _print_contexts(case.system_response.contexts, show_summary=False)
        _print_poisoned_context_hit_rate(
            hits=poisoned_hits,
            total=poisoned_total,
            show_summary=False,
        )
        _print_adversarial_texts(case.adversarial_texts, show_summary=False)


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


def _print_case_statistics(
    *,
    evaluation: Any,
    contexts: Sequence[str],
    poisoned_hits: int,
    poisoned_total: int,
    adversarial_texts: Sequence[str],
) -> None:
    ratio_percent = poisoned_hits / poisoned_total * 100.0 if poisoned_total else 0.0
    print("  Stats:")
    _print_field("Evaluation", _format_evaluation(evaluation), indent="    ")
    _print_field("Retrieved contexts", len(contexts), indent="    ")
    _print_field(
        "Poisoned context hit rate",
        f"{poisoned_hits}/{poisoned_total} ({ratio_percent:.2f}%)",
        indent="    ",
    )
    _print_field(
        "Generated adversarial contexts", len(adversarial_texts), indent="    "
    )


def _print_contexts(contexts: Sequence[str], *, show_summary: bool = True) -> None:
    if show_summary:
        print(f"  Retrieved contexts: {len(contexts)}")
    for context_index, context in enumerate(contexts, start=1):
        _print_field(
            f"Context {context_index}",
            context,
            indent="    ",
            limit=CONTEXT_TEXT_LIMIT,
        )


def _print_poisoned_context_hit_rate(
    *,
    hits: int,
    total: int,
    show_summary: bool = True,
) -> None:
    ratio_percent = (hits / total * 100.0) if total else 0.0
    if show_summary:
        print(f"  Poisoned context hit rate: {hits}/{total} ({ratio_percent:.2f}%)")


def _count_poisoned_context_hits(
    contexts: Sequence[str],
    adversarial_texts: Sequence[str],
) -> tuple[int, int]:
    normalized_adversarial = tuple(
        _normalize_text_for_match(adversarial_text)
        for adversarial_text in adversarial_texts
    )
    normalized_adversarial = tuple(
        adversarial_text
        for adversarial_text in normalized_adversarial
        if adversarial_text
    )
    if not normalized_adversarial:
        return 0, len(contexts)
    hits = sum(
        1
        for context in contexts
        if any(
            adversarial_text in _normalize_text_for_match(context)
            for adversarial_text in normalized_adversarial
        )
    )
    return hits, len(contexts)


def _print_adversarial_texts(
    adversarial_texts: Sequence[str], *, show_summary: bool = True
) -> None:
    if not adversarial_texts:
        _print_field("Generated adversarial contexts", "not provided")
        return
    if len(adversarial_texts) == 1:
        _print_field(
            "Generated adversarial context",
            adversarial_texts[0],
            limit=CONTEXT_TEXT_LIMIT,
        )
        return
    if show_summary:
        print(f"  Generated adversarial contexts: {len(adversarial_texts)}")
    for index, adversarial_text in enumerate(adversarial_texts, start=1):
        _print_field(
            f"Adversarial context {index}",
            adversarial_text,
            indent="    ",
            limit=CONTEXT_TEXT_LIMIT,
        )


def _normalize_text_for_match(text: str) -> str:
    return " ".join(text.casefold().split())


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
