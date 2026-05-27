import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from vexrag.cli.presentation.output_mode import OutputMode
from vexrag.core.scan.contracts import ScanCaseReport, ScanReport
from vexrag.core.scan.execution import (
    AttackChainScanReport,
    format_chain_step_summary_lines,
)

FIELD_TEXT_LIMIT = 2_000
CONTEXT_TEXT_LIMIT = 4_000


def print_scan_report(
    report: ScanReport,
    *,
    output_mode: OutputMode,
    show_raw_responses: bool = False,
    cases_emitted_during_run: bool = False,
) -> None:
    show_header = output_mode is not OutputMode.QUIET and not cases_emitted_during_run
    if show_header:
        print("VexRAG Scan")

    if (
        output_mode is not OutputMode.QUIET
        and isinstance(report, AttackChainScanReport)
        and len(report.step_reports) > 1
    ):
        if cases_emitted_during_run:
            print()
        print("Per-step summary:")
        for line in format_chain_step_summary_lines(report.step_reports):
            print(line)
        print()
    elif cases_emitted_during_run and output_mode is not OutputMode.QUIET:
        print()

    if output_mode is OutputMode.DETAILED and not cases_emitted_during_run:
        print_case_details(report.cases, show_raw_responses=show_raw_responses)

    warnings = collect_warnings(report)
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if output_mode is not OutputMode.QUIET:
        print()
        if output_mode is OutputMode.DETAILED:
            print(f"Cases: {report.total_cases}")
            print()
        elif output_mode is OutputMode.COMPACT:
            print(f"Cases: {report.total_cases}")
            print()

    verdict_label = "Final verdict" if output_mode is OutputMode.COMPACT else "Verdict"
    asr_label = "Success rate" if output_mode is OutputMode.DETAILED else "ASR"
    print_report_summary(report, verdict_label=verdict_label, asr_label=asr_label)


def print_report_summary(
    report: ScanReport,
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


def print_case_details(
    cases: Sequence[ScanCaseReport],
    *,
    show_raw_responses: bool = False,
) -> None:
    if not cases:
        return

    print("Case details:")
    for case_index, case in enumerate(cases, start=1):
        print_single_case_detail(
            case_index,
            case,
            show_raw_responses=show_raw_responses,
        )


def print_single_case_detail(
    case_index: int,
    case: ScanCaseReport,
    *,
    show_raw_responses: bool = False,
) -> None:
    label = case.case_id or f"#{case_index}"
    if not case.evaluation.completed:
        verdict = "NOT_EVALUATED"
    elif case.successful:
        verdict = "SUCCESS"
    else:
        verdict = "FAILED"
    poisoned_hits, poisoned_total = count_poisoned_context_hits(
        case.system_response.contexts,
        case.adversarial_texts,
    )
    print(flush=True)
    print(
        f"  Case {case_index}: {label} (run {case.run_index}) - {verdict}", flush=True
    )
    print_case_statistics(
        evaluation=case.evaluation,
        contexts=case.system_response.contexts,
        poisoned_hits=poisoned_hits,
        poisoned_total=poisoned_total,
        adversarial_texts=case.adversarial_texts,
    )
    print_field("Query", case.query)
    print_field("Expected incorrect answer", case.expected_incorrect_answer)
    print_field("LLM answer", case.system_response.answer)
    if case.evaluation.reason:
        print_field("Evaluation reason", case.evaluation.reason)
    if show_raw_responses and case.evaluation.raw is not None:
        print_field(
            "Judge raw response",
            format_raw_response(case.evaluation.raw),
        )
    print_contexts(case.system_response.contexts, show_summary=False)
    print_poisoned_context_hit_rate(
        hits=poisoned_hits,
        total=poisoned_total,
        show_summary=False,
    )
    print_adversarial_texts(case.adversarial_texts, show_summary=False)
    sys.stdout.flush()


def collect_warnings(report: ScanReport) -> tuple[str, ...]:
    warnings: list[str] = []
    warnings.extend(report.warnings)
    for case in report.cases:
        warnings.extend(case.warnings)
    return tuple(dict.fromkeys(warnings))


def print_case_statistics(
    *,
    evaluation: Any,
    contexts: Sequence[str],
    poisoned_hits: int,
    poisoned_total: int,
    adversarial_texts: Sequence[str],
) -> None:
    ratio_percent = poisoned_hits / poisoned_total * 100.0 if poisoned_total else 0.0
    print("  Stats:")
    print_field("Evaluation", format_evaluation(evaluation), indent="    ")
    print_field("Retrieved contexts", len(contexts), indent="    ")
    print_field(
        "Poisoned context hit rate",
        f"{poisoned_hits}/{poisoned_total} ({ratio_percent:.2f}%)",
        indent="    ",
    )
    print_field("Generated adversarial contexts", len(adversarial_texts), indent="    ")


def print_contexts(contexts: Sequence[str], *, show_summary: bool = True) -> None:
    if show_summary:
        print(f"  Retrieved contexts: {len(contexts)}")
    for context_index, context in enumerate(contexts, start=1):
        print_field(
            f"Context {context_index}",
            context,
            indent="    ",
            limit=CONTEXT_TEXT_LIMIT,
        )


def print_poisoned_context_hit_rate(
    *,
    hits: int,
    total: int,
    show_summary: bool = True,
) -> None:
    ratio_percent = (hits / total * 100.0) if total else 0.0
    if show_summary:
        print(f"  Poisoned context hit rate: {hits}/{total} ({ratio_percent:.2f}%)")


def print_adversarial_texts(
    adversarial_texts: Sequence[str], *, show_summary: bool = True
) -> None:
    if not adversarial_texts:
        print_field("Generated adversarial contexts", "not provided")
        return
    if len(adversarial_texts) == 1:
        print_field(
            "Generated adversarial context",
            adversarial_texts[0],
            limit=CONTEXT_TEXT_LIMIT,
        )
        return
    if show_summary:
        print(f"  Generated adversarial contexts: {len(adversarial_texts)}")
    for index, adversarial_text in enumerate(adversarial_texts, start=1):
        print_field(
            f"Adversarial context {index}",
            adversarial_text,
            indent="    ",
            limit=CONTEXT_TEXT_LIMIT,
        )


def count_poisoned_context_hits(
    contexts: Sequence[str],
    adversarial_texts: Sequence[str],
) -> tuple[int, int]:
    normalized_adversarial = tuple(
        normalize_text_for_match(adversarial_text)
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
            adversarial_text in normalize_text_for_match(context)
            for adversarial_text in normalized_adversarial
        )
    )
    return hits, len(contexts)


def format_evaluation(evaluation: Any) -> str:
    parts = [str(evaluation.strategy)]
    if evaluation.scores:
        scores = ", ".join(
            f"{name}={score:.4f}" for name, score in evaluation.scores.items()
        )
        parts.append(scores)
    judge_answer_label = judge_answer_label_for(evaluation)
    if judge_answer_label:
        parts.append(f"judge_answer_label={judge_answer_label}")
    return "; ".join(parts)


def judge_answer_label_for(evaluation: Any) -> object | None:
    judge = getattr(evaluation, "judge", None)
    if judge is not None:
        return getattr(judge, "label", None)
    return None


def format_raw_response(raw_response: object) -> str:
    if isinstance(raw_response, Mapping):
        return json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    return str(raw_response)


def normalize_text_for_match(text: str) -> str:
    return " ".join(text.casefold().split())


def print_field(
    label: str,
    value: object,
    *,
    indent: str = "  ",
    limit: int = FIELD_TEXT_LIMIT,
) -> None:
    text = truncate_text(str(value).strip(), limit=limit)
    if "\n" not in text and len(text) <= 120:
        print(f"{indent}{label}: {text}")
        return

    print(f"{indent}{label}:")
    for line in text.splitlines() or [""]:
        print(f"{indent}  {line}")


def truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}... [truncated {omitted} chars]"
