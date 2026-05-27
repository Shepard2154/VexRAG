import argparse
import logging

from vexrag.cli.logging_setup import configure_logging
from vexrag.usecases.config_io import load_config
from vexrag.usecases.generate_cases import run_generate_cases

LOGGER = logging.getLogger("vexrag.cli")


def run(args: argparse.Namespace) -> int:
    configure_logging(quiet=args.quiet, debug=args.debug)
    LOGGER.info("Loading generation config: %s", args.config)
    config = load_config(args.config)
    run_generate_cases(
        config,
        attack=str(args.attack),
        output=args.output,
        count=int(args.count),
        topic=args.topic,
        target_style=str(args.target_style),
        adv_per_query=args.adv_per_query,
        seed=args.seed,
        overwrite=bool(args.overwrite),
        quiet=bool(args.quiet),
    )
    return 0
