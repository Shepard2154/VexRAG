import argparse
import logging

from vexrag.cli.logging_setup import configure_logging
from vexrag.cli.presentation.doctor_report import print_doctor_result
from vexrag.usecases.config_io import load_config
from vexrag.usecases.doctor import run_doctor

LOGGER = logging.getLogger("vexrag.cli")


def run(args: argparse.Namespace) -> int:
    configure_logging(quiet=False, debug=args.debug)
    LOGGER.info("Loading doctor config: %s", args.config)
    config = load_config(args.config)
    result = run_doctor(
        config,
        base_dir=args.config.parent,
        check_llms=bool(args.check_llms),
    )
    print_doctor_result(result)
    return 0 if result.passed else 1
