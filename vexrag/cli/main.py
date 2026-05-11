import argparse
import json
import logging
import socket
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from vexrag.attack_algorithms.hijackrag.plugin import HIJACK_PLUGIN
from vexrag.attack_algorithms.poisonedrag.plugin import POISON_PLUGIN
from vexrag.cli.errors import CLIConfigError
from vexrag.cli.scan_builder import (
    build_scan_command,
    materialize_generate_cases_config,
    resolve_generate_cases_attack,
)
from vexrag.core.attacks import GenerateCasesParams
from vexrag.core.attacks.chain_command import (
    AttackChainScanReport,
    format_chain_step_summary_lines,
)
from vexrag.core.attacks.command import (
    ScanCaseReportProtocol,
    ScanReportProtocol,
)
from vexrag.core.attacks.registry import AttackRegistry
from vexrag.core.config import ScanConfigError
from vexrag.core.config.merge import deep_merge_mappings

LOGGER = logging.getLogger("vexrag.cli")
FIELD_TEXT_LIMIT = 2_000
CONTEXT_TEXT_LIMIT = 4_000


def _distribution_version() -> str:
    """Installed wheel/sdist version, or ``pyproject.toml`` when running from a checkout."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("vexrag")
    except PackageNotFoundError:
        pass
    try:
        repo_root = Path(__file__).resolve().parents[2]
        pyproject = repo_root / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (KeyError, OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return "0.0.0+unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ScanConfigError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vx",
        description="VexRAG command-line scanner.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"VexRAG {_distribution_version()}",
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
    scan.add_argument(
        "--detailed",
        action="store_true",
        help="Print per-case details instead of compact summary output.",
    )
    scan.set_defaults(handler=_run_scan)

    doctor = subcommands.add_parser(
        "doctor",
        help="Run environment and config preflight checks.",
    )
    _add_config_argument(doctor)
    doctor.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs.",
    )
    doctor.set_defaults(handler=_run_doctor)

    generate_cases = subcommands.add_parser(
        "generate-cases",
        help="Generate PoisonedRAG or HijackRAG cases YAML via the scan config LLM.",
    )
    _add_config_argument(generate_cases)
    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    generate_cases_attack_choices = ("auto", *registry.ids())
    generate_cases.add_argument(
        "--attack",
        choices=generate_cases_attack_choices,
        default="auto",
        metavar="NAME",
        help=(
            "Registered attack id or auto (only when the YAML has a single attacks step). "
            f"Built-in ids: {', '.join(registry.ids())}."
        ),
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
    if args.debug:
        config = deep_merge_mappings(
            config,
            {"scan": {"debug_include_raw_target_response": True}},
        )
    _log_config_summary(config)
    _run_preflight_checks(config)
    LOGGER.info("Building scan command")
    command = build_scan_command(config, base_dir=args.config.parent)
    requests = getattr(command, "requests", ())
    if requests:
        LOGGER.info("Loaded %d scan case(s)", len(requests))
    LOGGER.info("Running scan")
    stream_cases = not args.quiet and args.detailed
    if stream_cases:
        print("VexRAG Scan", flush=True)
        print(flush=True)
        print("Case details:", flush=True)

    case_ordinal: list[int] = [0]

    def _on_case_complete(case: ScanCaseReportProtocol) -> None:
        case_ordinal[0] += 1
        _print_single_case_detail(
            case_ordinal[0],
            case,
            show_raw_responses=args.debug,
        )

    report = command.run(
        on_case_complete=_on_case_complete if stream_cases else None,
    )
    LOGGER.info(
        "Scan complete: verdict=%s, success_rate=%.2f%% (%d successes / "
        "%d evaluated of %d case runs)",
        report.verdict.value.upper(),
        report.success_rate * 100,
        report.successful_cases,
        report.evaluated_cases,
        report.total_cases,
    )
    _print_report(
        report,
        show_raw_responses=args.debug,
        cases_emitted_during_run=stream_cases,
        compact=not args.detailed,
    )
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    _configure_logging(quiet=False, debug=args.debug)
    LOGGER.info("Loading doctor config: %s", args.config)
    config = _load_config(args.config)
    checks: tuple[tuple[str, Any], ...] = (
        ("target API availability", lambda: _preflight_target_system(config)),
        ("Ollama endpoint + required models", lambda: _preflight_ollama_models(config)),
        (
            "scan config validity",
            lambda: build_scan_command(config, base_dir=args.config.parent),
        ),
    )
    results: list[tuple[str, bool, str | None]] = []
    for name, check in checks:
        try:
            check()
        except (ScanConfigError, CLIConfigError, ValueError, RuntimeError) as exc:
            results.append((name, False, str(exc)))
        else:
            results.append((name, True, None))

    print("VexRAG Doctor")
    print()
    for name, ok, error in results:
        icon = "OK" if ok else "FAIL"
        print(f"[{icon}] {name}")
        if error:
            print(f"  -> {error}")

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"Doctor verdict: FAIL ({len(failed)} check(s) failed)")
        return 1
    print("Doctor verdict: PASS")
    return 0


def _run_generate_cases(args: argparse.Namespace) -> int:
    _configure_logging(quiet=args.quiet, debug=args.debug)
    LOGGER.info("Loading generation config: %s", args.config)
    config = _load_config(args.config)
    explicit = None if args.attack == "auto" else str(args.attack).strip().lower()

    attack_kind = resolve_generate_cases_attack(config, explicit=explicit)
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if output_path.exists() and not args.overwrite:
        raise CLIConfigError(
            f"output file already exists: {output_path}. Use --overwrite to replace it."
        )

    registry = AttackRegistry()
    registry.register(HIJACK_PLUGIN)
    registry.register(POISON_PLUGIN)
    plugin = registry.get(attack_kind)
    adv = max(1, int(args.adv_per_query))
    params = GenerateCasesParams(
        count=int(args.count),
        topic=args.topic,
        target_style=str(args.target_style),
        seed=args.seed,
        adv_per_query=adv,
    )
    LOGGER.info(
        "Generating %d case(s) via %s [target_style=%s]",
        params.count,
        plugin.display_name,
        params.target_style,
    )
    gen_config = materialize_generate_cases_config(
        config,
        attack_id=attack_kind,
    )
    cases = plugin.generate_cases(gen_config, params)
    payload = {"cases": [plugin.serialize_case_for_yaml(case) for case in cases]}
    yaml_attack = plugin.attack_id

    _write_yaml(output_path, payload)
    if args.quiet:
        print(output_path)
        return 0

    print(f"Generated {plugin.display_name} cases")
    print(f"Output: {output_path}")
    print(f"Cases: {len(cases)}")
    if args.topic:
        print(f"Topic: {args.topic}")
    print()
    print("Use in config:")
    print(
        f"  attacks: [ {{ id: {yaml_attack}, params: {{ case_files: ['{output_path}'] }} }} ]"
    )
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

    attacks = config.get("attacks")
    if isinstance(attacks, list) and attacks:
        ids: list[str] = []
        for item in attacks:
            if isinstance(item, Mapping):
                raw_id = item.get("id")
                if isinstance(raw_id, str) and raw_id.strip():
                    ids.append(raw_id.strip())
        if ids:
            LOGGER.info("Attacks (%d): %s", len(ids), ", ".join(ids))

    evaluations = config.get("evaluations")
    if isinstance(evaluations, Mapping):
        raw_list = evaluations.get("evaluators")
        n = len(raw_list) if isinstance(raw_list, list) else 0
        combine = evaluations.get("combine", "any")
        LOGGER.info("Evaluations: %d evaluator(s), combine=%s", n, combine)

    evaluation = config.get("evaluation")
    if isinstance(evaluation, Mapping) and not isinstance(evaluations, Mapping):
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


def _print_report(
    report: ScanReportProtocol,
    *,
    show_raw_responses: bool = False,
    cases_emitted_during_run: bool = False,
    compact: bool = True,
) -> None:
    if not cases_emitted_during_run:
        print("VexRAG Scan")
    if isinstance(report, AttackChainScanReport) and len(report.step_reports) > 1:
        if cases_emitted_during_run:
            print()
        print("Per-step summary:")
        for line in format_chain_step_summary_lines(report.step_reports):
            print(line)
        print()
    elif cases_emitted_during_run:
        print()
    if compact:
        _print_report_summary(report, verdict_label="Verdict", asr_label="ASR")
    else:
        _print_report_summary(report, verdict_label="Verdict", asr_label="Success rate")
        print(f"Cases: {report.total_cases}")
    print()
    if not compact and not cases_emitted_during_run:
        _print_case_details(report.cases, show_raw_responses=show_raw_responses)

    warnings = _collect_warnings(report)
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print()
    if compact:
        print(f"Cases: {report.total_cases}")
    _print_report_summary(report, verdict_label="Final verdict", asr_label="ASR")


def _run_preflight_checks(config: Mapping[str, Any]) -> None:
    _preflight_target_system(config)
    _preflight_ollama_models(config)


def _preflight_target_system(config: Mapping[str, Any]) -> None:
    target_http = _target_http_config(config)
    if not target_http:
        return
    base_url = target_http.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        return
    parsed = urlparse(base_url.strip())
    host = parsed.hostname
    port = parsed.port
    if host is None:
        return
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        route = str(target_http.get("route", "")).strip()
        hint = f"{base_url.rstrip('/')}/{route.lstrip('/')}" if route else base_url
        raise CLIConfigError(
            "target_system is unreachable. "
            f"Could not connect to {host}:{port} ({exc}). "
            f"Start your RAG service first (configured endpoint: {hint})."
        ) from exc


def _preflight_ollama_models(config: Mapping[str, Any]) -> None:
    required_models = _collect_ollama_models(config)
    for base_url, models in required_models.items():
        if not models:
            continue
        available = _fetch_ollama_models(base_url)
        missing = sorted(
            model for model in models if not _model_available(model, available)
        )
        if missing:
            raise CLIConfigError(
                "ollama model(s) not available at "
                f"{base_url}: {', '.join(missing)}. "
                "Pull them first (for example: ollama pull <model>)."
            )


def _collect_ollama_models(config: Mapping[str, Any]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    def _visit(node: Any) -> None:
        if isinstance(node, Mapping):
            provider = node.get("provider")
            base_url = node.get("base_url")
            model = node.get("model")
            if (
                isinstance(provider, str)
                and provider.strip().lower() == "ollama"
                and isinstance(base_url, str)
                and base_url.strip()
                and isinstance(model, str)
                and model.strip()
            ):
                found.setdefault(base_url.strip().rstrip("/"), set()).add(model.strip())
            for value in node.values():
                _visit(value)
            return
        if isinstance(node, list | tuple):
            for item in node:
                _visit(item)

    _visit(config)
    return found


def _fetch_ollama_models(base_url: str) -> set[str]:
    request = Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError) as exc:
        raise CLIConfigError(
            "could not query Ollama model list at "
            f"{base_url}/api/tags ({exc}). "
            "Ensure Ollama is running and reachable."
        ) from exc
    models_raw = payload.get("models")
    if not isinstance(models_raw, list):
        raise CLIConfigError(
            f"unexpected Ollama /api/tags response at {base_url}: missing models list"
        )
    names: set[str] = set()
    for model_info in models_raw:
        if not isinstance(model_info, Mapping):
            continue
        name = model_info.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _model_available(required: str, available: set[str]) -> bool:
    if required in available:
        return True
    if ":" in required:
        return False
    prefix = f"{required}:"
    return any(candidate.startswith(prefix) for candidate in available)


def _print_report_summary(
    report: ScanReportProtocol,
    *,
    verdict_label: str,
    asr_label: str,
) -> None:
    print(f"{verdict_label}: {report.verdict.value.upper()}")
    tail = ""
    if report.evaluated_cases != report.total_cases:
        tail = f", {report.total_cases} case runs total"
    print(
        f"{asr_label}: "
        f"{report.success_rate:.2%} "
        f"({report.successful_cases}/{report.evaluated_cases} evaluated{tail})"
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
        _print_single_case_detail(
            case_index,
            case,
            show_raw_responses=show_raw_responses,
        )


def _print_single_case_detail(
    case_index: int,
    case: ScanCaseReportProtocol,
    *,
    show_raw_responses: bool = False,
) -> None:
    label = case.case_id or f"#{case_index}"
    if not case.evaluation.evaluation_completed:
        verdict = "NOT_EVALUATED"
    elif case.successful:
        verdict = "SUCCESS"
    else:
        verdict = "FAILED"
    poisoned_hits, poisoned_total = _count_poisoned_context_hits(
        case.system_response.contexts,
        case.adversarial_texts,
    )
    print(flush=True)
    print(
        f"  Case {case_index}: {label} (run {case.run_index}) - {verdict}", flush=True
    )
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
    sys.stdout.flush()


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
