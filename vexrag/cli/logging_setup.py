import logging
import sys


def configure_logging(*, quiet: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
