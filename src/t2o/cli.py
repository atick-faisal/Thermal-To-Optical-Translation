"""``t2o`` command line interface.

Subcommands and the flag→config-path override table land in M0.8, once the config and
engine layers exist. This module is the sole caller of ``logging.basicConfig`` (PLAN.md
§13); library modules only ever take ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from t2o import __version__

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t2o", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    # Not `required=True` yet: with no subcommands registered, requiring one would make
    # `t2o --version` unreachable. Flip it in M0.8 when the first subcommand is added.
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )
    logger.error("no subcommands yet; the engine layer lands in M0.8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
