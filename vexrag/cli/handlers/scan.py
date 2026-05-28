import argparse
import logging
from collections.abc import Mapping

from vexrag.cli.logging_setup import configure_logging
from vexrag.cli.presentation.output_mode import OutputMode, resolve_output_mode
from vexrag.cli.presentation.scan_report import (
    print_scan_report,
    print_single_case_detail,
)
from vexrag.core.scan.config import deep_merge_mappings
from vexrag.core.scan.contracts import ScanCaseReport
from vexrag.usecases.config_io import load_config
from vexrag.usecases.preflight import (
    preflight_ollama_models,
    preflight_target_system,
    preflight_vllm_models,
)
from vexrag.usecases.scan_service import (
    build_scan_command,
    materialize_preflight_config,
    run_scan,
)

LOGGER = logging.getLogger("vexrag.cli")


def run(args: argparse.Namespace) -> int:
    output_mode = resolve_output_mode(quiet=args.quiet, detailed=args.detailed)
    configure_logging(quiet=output_mode is OutputMode.QUIET, debug=args.debug)
    attack = None if args.attack is None else str(args.attack).strip().lower()
    LOGGER.info("Loading scan config: %s", args.config)
    config = load_config(args.config)
    if args.debug:
        config = deep_merge_mappings(
            config,
            {"scan": {"debug_include_raw_target_response": True}},
        )
    log_config_summary(config)
    preflight_config = materialize_preflight_config(config, attack_id=attack)
    preflight_target_system(preflight_config)
    preflight_ollama_models(preflight_config)
    preflight_vllm_models(preflight_config)
    LOGGER.info("Building scan command")
    command = build_scan_command(
        config,
        base_dir=args.config.parent,
        attack=attack,
        probe_llms=True,
    )
    requests = getattr(command, "requests", ())
    if requests:
        LOGGER.info("Loaded %d scan case(s)", len(requests))
    LOGGER.info("Running scan")
    stream_cases = output_mode is OutputMode.DETAILED
    if stream_cases:
        print("VexRAG Scan", flush=True)
        print(flush=True)
        print("Case details:", flush=True)

    case_ordinal: list[int] = [0]

    def _on_case_complete(case: ScanCaseReport) -> None:
        case_ordinal[0] += 1
        print_single_case_detail(
            case_ordinal[0],
            case,
            show_raw_responses=args.debug,
        )

    report = run_scan(
        command,
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
    print_scan_report(
        report,
        output_mode=output_mode,
        show_raw_responses=args.debug,
        cases_emitted_during_run=stream_cases,
    )
    return 0


def log_config_summary(config: Mapping) -> None:
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
                    ids.append(raw_id.strip().lower())
        if ids:
            LOGGER.info("Attacks (%d): %s", len(ids), ", ".join(ids))

    evaluation = config.get("evaluation")
    if isinstance(evaluation, Mapping):
        strategy = evaluation.get("strategy", "embedding_similarity")
        LOGGER.info("Evaluation: %s", strategy)


def _target_http_config(config: Mapping) -> Mapping | None:
    target = config.get("target_system")
    if not isinstance(target, Mapping):
        return None
    http_config = target.get("http", target)
    if not isinstance(http_config, Mapping):
        return None
    return http_config
