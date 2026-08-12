"""Convert `dataset/raw/<name>/` layouts into the internal representation the training code
reads (PLAN.md §9): `dataset/processed/<name>/{split}/{visible,infrared}/{images,labels}` +
`data.yaml`.

Companion to `fetch_datasets.py` -- that script only fetches; nothing there converts a raw
layout, this is that step. Only datasets that are genuinely paired belong here; CPLID
(RGB-only) and HIT-UAV (IR-only) don't fit the paired contract and have no adapter.

FLIR-aligned's adapter reads directly out of `aligned.zip` (never extracted to disk) and can
take upwards of a minute on the real ~1.4GB archive -- expected, not a hang.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the
way `fetch_datasets.py` does for the same reason.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from t2o.data.adapters import adapt_flir, adapt_msrs

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

DEFAULT_RAW_ROOT = Path("dataset/raw")
DEFAULT_DEST_ROOT = Path("dataset/processed")

# One entry per adapted dataset. Registry, not a chain of ifs, so adding the next one is a
# one-line addition here plus its own `adapters/<name>.py`.
ADAPTERS: dict[str, Callable[[Path, Path], Path]] = {
    "msrs": adapt_msrs,
    "flir": adapt_flir,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=[*sorted(ADAPTERS), "all"],
        help="which dataset(s) to adapt (default: all)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"root `fetch_datasets.py` populated (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"destination root (default: {DEFAULT_DEST_ROOT})",
    )
    return parser


def resolve_names(names: Sequence[str]) -> list[str]:
    return sorted(ADAPTERS) if "all" in names else list(names)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    for name in resolve_names(args.dataset):
        data_yaml = ADAPTERS[name](args.raw_root / name, args.dest_root / name)
        logger.info("%s -> %s", name, data_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
