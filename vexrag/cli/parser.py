import argparse
import tomllib
from pathlib import Path

from vexrag.attack_algorithms.registries import create_attack_method_registry
from vexrag.cli.handlers import doctor as doctor_handler
from vexrag.cli.handlers import generate_cases as generate_cases_handler
from vexrag.cli.handlers import scan as scan_handler


def distribution_version() -> str:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vx",
        description="VexRAG command-line scanner.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"VexRAG {distribution_version()}",
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
        help="Only print the final scan report (warnings still shown).",
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
    registry = create_attack_method_registry()
    scan.add_argument(
        "--attack",
        choices=registry.ids(),
        default=None,
        metavar="ID",
        help="Run only the attack step with this id (for multi-attack configs).",
    )
    scan.set_defaults(handler=scan_handler.run)

    doctor = subcommands.add_parser(
        "doctor",
        help="Run environment and config preflight checks.",
    )
    _add_config_argument(doctor)
    doctor.add_argument(
        "--check-llms",
        action="store_true",
        help="Probe configured attack and judge LLMs (slower; requires live endpoints).",
    )
    doctor.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs.",
    )
    doctor.set_defaults(handler=doctor_handler.run)

    generate_cases = subcommands.add_parser(
        "generate-cases",
        help="Generate PoisonedRAG or HijackRAG cases YAML via the scan config LLM.",
    )
    _add_config_argument(generate_cases)
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
        default=None,
        metavar="N",
        help=(
            "adv_per_query per case in the YAML. "
            "Default when omitted: hijackrag=1, poisonedrag=5."
        ),
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
    generate_cases.set_defaults(handler=generate_cases_handler.run)

    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML scan config.",
    )
